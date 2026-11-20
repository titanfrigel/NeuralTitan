#include "runtime/trainer.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "nn/loss.hpp"
#include "nn/module.hpp"
#include "ops/elementwise.hpp"
#include "optim/clip.hpp"
#include "optim/schedule.hpp"

namespace titan::runtime {

namespace {

// ---- train config parsing -------------------------------------------------

constexpr double kMaxExactInt = 9007199254740992.0;  // 2^53

[[noreturn]] void config_error(const std::string& path,
                               const std::string& message) {
    throw_error(__FILE__, __LINE__, "train config: " + path + ": " + message);
}

const char* kind_name(const io::JsonValue& v) {
    if (v.is_null()) return "null";
    if (v.is_bool()) return "a bool";
    if (v.is_number()) return "a number";
    if (v.is_string()) return "a string";
    if (v.is_array()) return "an array";
    return "an object";
}

std::string field_path(std::string_view key) {
    return "train." + std::string(key);
}

// An integral JSON number >= min_value.
std::int64_t read_int(const io::JsonValue& v, std::string_view key,
                      std::int64_t min_value) {
    const std::string path = field_path(key);
    if (!v.is_number()) {
        config_error(path,
                     std::string("expected an integer, got ") + kind_name(v));
    }
    const double d = v.as_number();
    if (d != std::trunc(d) || std::fabs(d) > kMaxExactInt) {
        config_error(path, "expected an integer, got " + std::to_string(d));
    }
    const auto n = static_cast<std::int64_t>(d);
    if (n < min_value) {
        config_error(path, "must be >= " + std::to_string(min_value) +
                               ", got " + std::to_string(n));
    }
    return n;
}

// A finite JSON number that fits in a float.
double read_number(const io::JsonValue& v, std::string_view key) {
    const std::string path = field_path(key);
    if (!v.is_number()) {
        config_error(path,
                     std::string("expected a number, got ") + kind_name(v));
    }
    const double d = v.as_number();
    if (!std::isfinite(d) || std::fabs(d) > std::numeric_limits<float>::max()) {
        config_error(path, "must be a finite float, got " + std::to_string(d));
    }
    return d;
}

// A float > 0 that does not round to zero.
float read_positive_float(const io::JsonValue& v, std::string_view key) {
    const double d = read_number(v, key);
    if (!(d > 0.0)) {
        config_error(field_path(key), "must be > 0, got " + std::to_string(d));
    }
    const auto f = static_cast<float>(d);
    if (f == 0.0f) {
        config_error(field_path(key), "is too small for a float");
    }
    return f;
}

// Reads `key` when present, otherwise keeps the default in `out`.
template <typename Reader>
void read_optional(const io::JsonValue& section, std::string_view key,
                   Reader reader) {
    if (const io::JsonValue* v = section.find(key)) {
        reader(*v);
    }
}

}  // namespace

TrainConfig parse_train_config(const io::JsonValue& section) {
    if (!section.is_object()) {
        config_error("train", std::string("expected an object, got ") +
                                  kind_name(section));
    }
    constexpr std::initializer_list<std::string_view> kKnown = {
        "batch",         "seq",          "steps",
        "grad_accum",    "lr",           "min_lr",
        "warmup",        "weight_decay", "beta1",
        "beta2",         "eps",          "grad_clip",
        "eval_interval", "eval_batches", "log_interval",
        "seed",          "val_fraction"};
    for (const auto& [key, member] : section.as_object()) {
        bool known = false;
        for (std::string_view k : kKnown) {
            known = known || key == k;
        }
        if (!known) {
            config_error(field_path(key), "unknown key");
        }
    }
    const auto require = [&](std::string_view key) -> const io::JsonValue& {
        const io::JsonValue* v = section.find(key);
        if (v == nullptr) {
            config_error(field_path(key), "missing required field");
        }
        return *v;
    };

    TrainConfig c;
    c.batch = read_int(require("batch"), "batch", 1);
    c.seq = read_int(require("seq"), "seq", 1);
    c.steps = read_int(require("steps"), "steps", 1);
    c.lr = read_positive_float(require("lr"), "lr");
    const double min_lr = read_number(require("min_lr"), "min_lr");
    if (min_lr < 0.0 || min_lr > static_cast<double>(c.lr)) {
        config_error(field_path("min_lr"),
                     "must be in [0, lr], got " + std::to_string(min_lr));
    }
    c.min_lr = static_cast<float>(min_lr);
    c.warmup = read_int(require("warmup"), "warmup", 1);
    if (c.warmup >= c.steps) {
        config_error(field_path("warmup"),
                     "must be < steps " + std::to_string(c.steps) + ", got " +
                         std::to_string(c.warmup));
    }
    c.eval_interval = read_int(require("eval_interval"), "eval_interval", 1);
    c.eval_batches = read_int(require("eval_batches"), "eval_batches", 1);
    c.log_interval = read_int(require("log_interval"), "log_interval", 1);
    c.seed = static_cast<std::uint64_t>(read_int(require("seed"), "seed", 0));

    read_optional(section, "grad_accum", [&](const io::JsonValue& v) {
        c.grad_accum = read_int(v, "grad_accum", 1);
    });
    read_optional(section, "grad_clip", [&](const io::JsonValue& v) {
        c.grad_clip = read_positive_float(v, "grad_clip");
    });
    read_optional(section, "eps", [&](const io::JsonValue& v) {
        c.eps = read_positive_float(v, "eps");
    });
    read_optional(section, "weight_decay", [&](const io::JsonValue& v) {
        const double d = read_number(v, "weight_decay");
        if (d < 0.0) {
            config_error(field_path("weight_decay"),
                         "must be >= 0, got " + std::to_string(d));
        }
        c.weight_decay = static_cast<float>(d);
    });
    for (const auto& [key, target] :
         {std::pair<std::string_view, float*>{"beta1", &c.beta1},
          std::pair<std::string_view, float*>{"beta2", &c.beta2}}) {
        read_optional(section, key, [&](const io::JsonValue& v) {
            const double d = read_number(v, key);
            if (d < 0.0 || d >= 1.0) {
                config_error(field_path(key),
                             "must be in [0, 1), got " + std::to_string(d));
            }
            *target = static_cast<float>(d);
        });
    }
    read_optional(section, "val_fraction", [&](const io::JsonValue& v) {
        const double d = read_number(v, "val_fraction");
        if (!(d > 0.0 && d < 1.0)) {
            config_error(field_path("val_fraction"),
                         "must be in (0, 1), got " + std::to_string(d));
        }
        c.val_fraction = static_cast<float>(d);
    });
    return c;
}

namespace {

std::string format_lr(float lr) {
    std::ostringstream out;
    out << std::scientific << std::setprecision(1) << lr;
    std::string s = out.str();
    // MSVC prints three exponent digits ("e-004"); keep two ("e-04").
    const std::size_t e = s.find('e');
    if (e != std::string::npos && s.size() - e == 5 && s[e + 2] == '0') {
        s.erase(e + 2, 1);
    }
    return s;
}

}  // namespace

Trainer::Trainer(model::Model& model, TrainConfig config,
                 std::vector<std::int32_t> train_tokens,
                 std::vector<std::int32_t> val_tokens, std::ostream& log)
    : model_(model),
      config_(config),
      train_loader_(std::move(train_tokens), config.batch, config.seq,
                    config.seed),
      val_loader_(std::move(val_tokens), config.batch, config.seq,
                  config.seed + 1),
      log_(log),
      optimizer_(model.parameters(),
                 optim::AdamWConfig{config.lr, config.beta1, config.beta2,
                                    config.eps, config.weight_decay}) {
    TITAN_CHECK(config.seq <= model.config().context_length,
                "Trainer: train.seq " + std::to_string(config.seq) +
                    " exceeds the model context_length " +
                    std::to_string(model.config().context_length));
}

float Trainer::evaluate() {
    nn::CrossEntropyLoss loss;
    double total = 0.0;
    for (std::int64_t i = 0; i < config_.eval_batches; ++i) {
        const DataLoader::Batch batch = val_loader_.next();
        nn::ForwardContext ctx;
        ctx.training = false;
        total += loss.forward(
            model_.forward(batch.inputs, config_.batch, config_.seq, ctx),
            batch.targets);
    }
    return static_cast<float>(total /
                              static_cast<double>(config_.eval_batches));
}

TrainResult Trainer::run() {
    using Clock = std::chrono::steady_clock;
    const std::vector<nn::Parameter*> params = model_.parameters();
    const std::int64_t tokens_per_step =
        config_.grad_accum * config_.batch * config_.seq;
    const float accum_scale = 1.0f / static_cast<float>(config_.grad_accum);

    TrainResult result;
    nn::CrossEntropyLoss loss_fn;
    const auto start = Clock::now();
    auto last_log = start;
    std::int64_t tokens_since_log = 0;

    for (std::int64_t step = 0; step < config_.steps; ++step) {
        const std::int64_t shown = step + 1;
        nn::zero_grad(params);
        // The mean over micro-batches of the per-micro-batch mean loss; each
        // micro-batch gradient is scaled by 1/grad_accum so the accumulated
        // gradient is that of this mean.
        double loss_sum = 0.0;
        for (std::int64_t micro = 0; micro < config_.grad_accum; ++micro) {
            const DataLoader::Batch batch = train_loader_.next();
            nn::ForwardContext ctx;
            const Tensor logits =
                model_.forward(batch.inputs, config_.batch, config_.seq, ctx);
            loss_sum += loss_fn.forward(logits, batch.targets);
            Tensor dlogits = loss_fn.backward();
            ops::scale_(dlogits, accum_scale);
            model_.backward(dlogits);
        }
        const auto loss = static_cast<float>(loss_sum / config_.grad_accum);
        const double grad_norm =
            optim::clip_grad_norm(params, config_.grad_clip);
        if (!std::isfinite(loss) || !std::isfinite(grad_norm)) {
            std::ostringstream msg;
            msg << "training diverged at step " << shown << ": loss=" << loss
                << " grad_norm=" << grad_norm;
            throw_error(__FILE__, __LINE__, msg.str());
        }
        const float lr = optim::lr_at(step, config_.warmup, config_.steps,
                                      config_.lr, config_.min_lr);
        optimizer_.step(lr);

        result.final_train_loss = loss;
        result.tokens += tokens_per_step;
        tokens_since_log += tokens_per_step;

        if (shown % config_.log_interval == 0) {
            const auto now = Clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - last_log).count();
            const double tok_per_s =
                elapsed > 0.0 ? static_cast<double>(tokens_since_log) / elapsed
                              : 0.0;
            log_ << "step " << shown << "/" << config_.steps << " loss "
                 << std::fixed << std::setprecision(4) << loss << " lr "
                 << format_lr(lr) << " grad_norm " << std::setprecision(3)
                 << grad_norm << " tok/s " << std::setprecision(0) << tok_per_s
                 << "\n";
            last_log = now;
            tokens_since_log = 0;
        }
        if (shown % config_.eval_interval == 0 || shown == config_.steps) {
            result.final_val_loss = evaluate();
            log_ << "eval step " << shown << " val_loss " << std::fixed
                 << std::setprecision(4) << result.final_val_loss << "\n";
        }
    }
    result.seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

}  // namespace titan::runtime
