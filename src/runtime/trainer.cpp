#include "runtime/trainer.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "io/json_fields.hpp"
#include "nn/loss.hpp"
#include "nn/module.hpp"
#include "ops/elementwise.hpp"
#include "optim/clip.hpp"
#include "optim/schedule.hpp"

namespace titan::runtime {

TrainConfig parse_train_config(const io::JsonValue& section) {
    const io::JsonFields f(
        section, "train config", "train",
        {"batch", "seq", "steps", "grad_accum", "lr", "min_lr", "warmup",
         "weight_decay", "beta1", "beta2", "eps", "grad_clip", "eval_interval",
         "eval_batches", "log_interval", "seed", "val_fraction",
         "checkpoint_interval"});

    TrainConfig c;
    c.batch = f.integer(f.require("batch"), "batch", 1);
    c.seq = f.integer(f.require("seq"), "seq", 1);
    c.steps = f.integer(f.require("steps"), "steps", 1);
    c.lr = f.positive_float(f.require("lr"), "lr");
    const double min_lr = f.number(f.require("min_lr"), "min_lr");
    if (min_lr < 0.0 || min_lr > static_cast<double>(c.lr)) {
        f.fail(f.path_of("min_lr"),
               "must be in [0, lr], got " + std::to_string(min_lr));
    }
    c.min_lr = static_cast<float>(min_lr);
    c.warmup = f.integer(f.require("warmup"), "warmup", 1);
    if (c.warmup >= c.steps) {
        f.fail(f.path_of("warmup"), "must be < steps " +
                                        std::to_string(c.steps) + ", got " +
                                        std::to_string(c.warmup));
    }
    c.eval_interval = f.integer(f.require("eval_interval"), "eval_interval", 1);
    c.eval_batches = f.integer(f.require("eval_batches"), "eval_batches", 1);
    c.log_interval = f.integer(f.require("log_interval"), "log_interval", 1);
    c.seed =
        static_cast<std::uint64_t>(f.integer(f.require("seed"), "seed", 0));

    if (const io::JsonValue* v = f.find("grad_accum")) {
        c.grad_accum = f.integer(*v, "grad_accum", 1);
    }
    if (const io::JsonValue* v = f.find("grad_clip")) {
        c.grad_clip = f.positive_float(*v, "grad_clip");
    }
    if (const io::JsonValue* v = f.find("eps")) {
        c.eps = f.positive_float(*v, "eps");
    }
    if (const io::JsonValue* v = f.find("weight_decay")) {
        const double d = f.number(*v, "weight_decay");
        if (d < 0.0) {
            f.fail(f.path_of("weight_decay"),
                   "must be >= 0, got " + std::to_string(d));
        }
        c.weight_decay = static_cast<float>(d);
    }
    const auto read_beta = [&](std::string_view key, float& out) {
        if (const io::JsonValue* v = f.find(key)) {
            const double d = f.number(*v, key);
            if (d < 0.0 || d >= 1.0) {
                f.fail(f.path_of(key),
                       "must be in [0, 1), got " + std::to_string(d));
            }
            out = static_cast<float>(d);
        }
    };
    read_beta("beta1", c.beta1);
    read_beta("beta2", c.beta2);
    if (const io::JsonValue* v = f.find("val_fraction")) {
        const double d = f.number(*v, "val_fraction");
        if (!(d > 0.0 && d < 1.0)) {
            f.fail(f.path_of("val_fraction"),
                   "must be in (0, 1), got " + std::to_string(d));
        }
        c.val_fraction = static_cast<float>(d);
    }
    if (const io::JsonValue* v = f.find("checkpoint_interval")) {
        c.checkpoint_interval = f.integer(*v, "checkpoint_interval", 0);
    }
    return c;
}

io::JsonValue train_config_to_json(const TrainConfig& c) {
    io::JsonValue o = io::JsonValue::object();
    const auto integer = [&](const char* key, std::int64_t v) {
        o.set(key, io::JsonValue::number(static_cast<double>(v)));
    };
    const auto real = [&](const char* key, float v) {
        o.set(key, io::JsonValue::number(static_cast<double>(v)));
    };
    integer("batch", c.batch);
    integer("seq", c.seq);
    integer("steps", c.steps);
    integer("grad_accum", c.grad_accum);
    real("lr", c.lr);
    real("min_lr", c.min_lr);
    integer("warmup", c.warmup);
    real("weight_decay", c.weight_decay);
    real("beta1", c.beta1);
    real("beta2", c.beta2);
    real("eps", c.eps);
    real("grad_clip", c.grad_clip);
    integer("eval_interval", c.eval_interval);
    integer("eval_batches", c.eval_batches);
    integer("log_interval", c.log_interval);
    integer("seed", static_cast<std::int64_t>(c.seed));
    real("val_fraction", c.val_fraction);
    integer("checkpoint_interval", c.checkpoint_interval);
    return o;
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

// Moves the model to `device` and returns its parameters, so that the
// optimizer, constructed from them, creates its moments on that device.
std::vector<nn::Parameter*> parameters_on(model::Model& model, Device device) {
    model.to(device);
    return model.parameters();
}

}  // namespace

Trainer::Trainer(model::Model& model, TrainConfig config,
                 std::vector<std::int32_t> train_tokens,
                 std::vector<std::int32_t> val_tokens, std::ostream& log)
    : Trainer(model, config, std::move(train_tokens), std::move(val_tokens),
              log, std::nullopt) {}

Trainer::Trainer(model::Model& model, TrainConfig config,
                 std::vector<std::int32_t> train_tokens,
                 std::vector<std::int32_t> val_tokens, std::ostream& log,
                 std::optional<TrainerState> resume)
    : model_(model),
      config_(config),
      train_tokens_(static_cast<std::int64_t>(train_tokens.size())),
      val_tokens_(static_cast<std::int64_t>(val_tokens.size())),
      train_loader_(std::move(train_tokens), config.batch, config.seq,
                    config.seed),
      val_loader_(std::move(val_tokens), config.batch, config.seq,
                  config.seed + 1),
      log_(log),
      optimizer_(parameters_on(model, config.device),
                 optim::AdamWConfig{config.lr, config.beta1, config.beta2,
                                    config.eps, config.weight_decay}) {
    TITAN_CHECK(config.grad_accum >= 1,
                "Trainer: grad_accum must be >= 1, got " +
                    std::to_string(config.grad_accum));
    TITAN_CHECK(config.log_interval >= 1,
                "Trainer: log_interval must be >= 1, got " +
                    std::to_string(config.log_interval));
    TITAN_CHECK(config.eval_interval >= 1,
                "Trainer: eval_interval must be >= 1, got " +
                    std::to_string(config.eval_interval));
    TITAN_CHECK(config.eval_batches >= 1,
                "Trainer: eval_batches must be >= 1, got " +
                    std::to_string(config.eval_batches));
    TITAN_CHECK(config.seq <= model.config().context_length,
                "Trainer: train.seq " + std::to_string(config.seq) +
                    " exceeds the model context_length " +
                    std::to_string(model.config().context_length));
    if (resume) {
        resume_from(*resume);
    }
}

void Trainer::resume_from(const TrainerState& s) {
    const auto changed = [](const char* which, std::int64_t saved,
                            std::int64_t now) {
        return std::string("resume: ") + which +
               " tokens changed (checkpoint " + std::to_string(saved) +
               ", now " + std::to_string(now) + ")";
    };
    TITAN_CHECK(s.train_tokens == train_tokens_,
                changed("train", s.train_tokens, train_tokens_));
    TITAN_CHECK(s.val_tokens == val_tokens_,
                changed("val", s.val_tokens, val_tokens_));
    TITAN_CHECK(s.step >= 0 && s.step <= config_.steps,
                "resume: step " + std::to_string(s.step) +
                    " is outside [0, train.steps " +
                    std::to_string(config_.steps) + "]");
    // One optimizer update per trainer step.
    TITAN_CHECK(s.optimizer.steps == s.step,
                "resume: optimizer step count " +
                    std::to_string(s.optimizer.steps) +
                    " differs from the step " + std::to_string(s.step));
    // The optimizer was built after the model moved to config.device, so
    // load_state copies the moments onto that device. A throw from here on
    // leaves no Trainer behind, so a partial restore is never observed.
    optimizer_.load_state(s.optimizer);
    train_loader_.rng().load_state(s.train_rng);
    val_loader_.rng().load_state(s.val_rng);
    step_ = s.step;
}

void Trainer::on_checkpoint(std::function<void(std::int64_t step)> callback) {
    on_checkpoint_ = std::move(callback);
}

TrainerState Trainer::state() const {
    TrainerState s;
    s.step = step_;
    s.optimizer = optimizer_.state();
    s.train_rng = train_loader_.rng().save_state();
    s.val_rng = val_loader_.rng().save_state();
    s.train_tokens = train_tokens_;
    s.val_tokens = val_tokens_;
    return s;
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

TrainResult Trainer::run(std::optional<std::int64_t> stop_after) {
    // Without stop_after, a run whose steps are all done runs nothing.
    const std::int64_t last = stop_after.value_or(config_.steps);
    TITAN_CHECK(!stop_after || (last > step_ && last <= config_.steps),
                "Trainer::run: stop_after " + std::to_string(last) +
                    " must be in [" + std::to_string(step_ + 1) + ", " +
                    std::to_string(config_.steps) + "]");
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

    // `step` is 0-based, as the schedule expects; `shown` is 1-based.
    for (std::int64_t step = step_; step < last; ++step) {
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
            std::ostringstream line;
            line << "step " << shown << "/" << config_.steps << " loss "
                 << std::fixed << std::setprecision(4) << loss << " lr "
                 << format_lr(lr) << " grad_norm " << std::setprecision(3)
                 << grad_norm << " tok/s " << std::setprecision(0) << tok_per_s
                 << "\n";
            log_ << line.str() << std::flush;
            last_log = now;
            tokens_since_log = 0;
        }
        if (shown % config_.eval_interval == 0 || shown == config_.steps) {
            result.final_val_loss = evaluate();
            std::ostringstream line;
            line << "eval step " << shown << " val_loss " << std::fixed
                 << std::setprecision(4) << result.final_val_loss << "\n";
            log_ << line.str() << std::flush;
        }
        step_ = shown;
        const bool periodic = config_.checkpoint_interval > 0 &&
                              shown % config_.checkpoint_interval == 0;
        if (on_checkpoint_ && (periodic || shown == last)) {
            on_checkpoint_(shown);
        }
    }
    result.seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

}  // namespace titan::runtime
