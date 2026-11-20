#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "harness/test.hpp"
#include "io/json.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "runtime/trainer.hpp"

using titan::Rng;
using titan::model::llama_style;
using titan::model::Model;
using titan::runtime::parse_train_config;
using titan::runtime::TrainConfig;
using titan::runtime::Trainer;

namespace {

const char* const kValidConfig = R"({
    "batch": 4, "seq": 8, "steps": 100, "lr": 3e-3, "min_lr": 3e-4,
    "warmup": 10, "eval_interval": 50, "eval_batches": 2,
    "log_interval": 20, "seed": 42
})";

TrainConfig small_config() {
    TrainConfig c = parse_train_config(titan::io::parse_json(kValidConfig));
    return c;
}

// 0 1 2 3 4 0 1 2 3 4 ...
std::vector<std::int32_t> repeating_tokens(std::size_t n) {
    std::vector<std::int32_t> t(n);
    for (std::size_t i = 0; i < n; ++i) {
        t[i] = static_cast<std::int32_t>(i % 5);
    }
    return t;
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

}  // namespace

TITAN_TEST(TrainConfig, ParsesAndRejectsMissingField) {
    const TrainConfig c = small_config();
    TITAN_EXPECT_EQ(c.batch, std::int64_t{4});
    TITAN_EXPECT_EQ(c.seq, std::int64_t{8});
    TITAN_EXPECT_EQ(c.seed, std::uint64_t{42});
    // Defaults.
    TITAN_EXPECT_EQ(c.grad_accum, std::int64_t{1});
    TITAN_EXPECT_EQ(c.grad_clip, 1.0f);
    TITAN_EXPECT_EQ(c.val_fraction, 0.1f);
    TITAN_EXPECT_EQ(c.beta1, 0.9f);
    TITAN_EXPECT_EQ(c.beta2, 0.95f);
    TITAN_EXPECT_EQ(c.eps, 1e-8f);
    TITAN_EXPECT_EQ(c.weight_decay, 0.1f);

    // A missing required field is named by its path.
    titan::io::JsonValue missing = titan::io::parse_json(kValidConfig);
    std::string message;
    try {
        titan::io::JsonValue::Object members;
        for (const auto& [key, value] : missing.as_object()) {
            if (key != "lr") {
                members.emplace_back(key, value);
            }
        }
        parse_train_config(titan::io::JsonValue::object(std::move(members)));
    } catch (const titan::Error& e) {
        message = e.what();
    }
    TITAN_EXPECT(contains(message, "train.lr"));
    TITAN_EXPECT(contains(message, "missing"));

    // An unknown key, a bad range and a wrong type are rejected with a path.
    titan::io::JsonValue unknown = titan::io::parse_json(kValidConfig);
    unknown.set("lr_decay", titan::io::JsonValue::number(1.0));
    TITAN_EXPECT_THROWS(parse_train_config(unknown), titan::Error);
    titan::io::JsonValue warm = titan::io::parse_json(kValidConfig);
    warm.set("warmup", titan::io::JsonValue::number(100.0));
    TITAN_EXPECT_THROWS(parse_train_config(warm), titan::Error);
    titan::io::JsonValue frac = titan::io::parse_json(kValidConfig);
    frac.set("val_fraction", titan::io::JsonValue::number(1.0));
    TITAN_EXPECT_THROWS(parse_train_config(frac), titan::Error);
    titan::io::JsonValue typed = titan::io::parse_json(kValidConfig);
    typed.set("batch", titan::io::JsonValue::string("4"));
    TITAN_EXPECT_THROWS(parse_train_config(typed), titan::Error);
    TITAN_EXPECT_THROWS(parse_train_config(titan::io::JsonValue::array()),
                        titan::Error);
}

TITAN_TEST(Trainer, LossFallsOnRepeatingPattern) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    Rng init(11);
    m.init(init);
    const TrainConfig c = small_config();
    std::ostringstream log;
    Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100), log);
    const titan::runtime::TrainResult r = trainer.run();
    TITAN_EXPECT(r.final_val_loss < 0.5f);
    TITAN_EXPECT_EQ(r.tokens, std::int64_t{100 * 4 * 8});
    const std::string text = log.str();
    TITAN_EXPECT(contains(text, "step 20/100 loss "));
    // The learning rate is printed as d.de-dd.
    const std::size_t lr_at = text.find(" lr ");
    TITAN_EXPECT(lr_at != std::string::npos);
    const std::string lr_text = text.substr(lr_at + 4, 8);
    TITAN_EXPECT(lr_text[1] == '.' && lr_text[3] == 'e' && lr_text[4] == '-');
    TITAN_EXPECT(lr_text[5] != ' ' && lr_text[6] != ' ' && lr_text[7] == ' ');
    TITAN_EXPECT(contains(text, " grad_norm "));
    TITAN_EXPECT(contains(text, " tok/s "));
    TITAN_EXPECT(contains(text, "eval step 50 val_loss "));
    TITAN_EXPECT(contains(text, "eval step 100 val_loss "));
}

TITAN_TEST(Trainer, StopsOnNonFiniteLoss) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    Rng init(11);
    m.init(init);
    TrainConfig c = small_config();
    c.lr = 1e30f;
    c.min_lr = 1e30f;
    c.steps = 30;
    c.warmup = 1;
    std::ostringstream log;
    Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100), log);
    std::string message;
    try {
        trainer.run();
    } catch (const titan::Error& e) {
        message = e.what();
    }
    TITAN_EXPECT(contains(message, "training diverged at step "));
    TITAN_EXPECT(contains(message, "loss="));
    TITAN_EXPECT(contains(message, "grad_norm="));
}

TITAN_TEST(Trainer, RejectsSeqLongerThanContext) {
    Model m(llama_style(5, 16, 1, 2, 1, 4));
    TrainConfig c = small_config();  // seq 8 > context 4
    std::ostringstream log;
    std::string message;
    try {
        Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100),
                        log);
    } catch (const titan::Error& e) {
        message = e.what();
    }
    TITAN_EXPECT(contains(message, "train.seq 8"));
    TITAN_EXPECT(contains(message, "context_length 4"));
}

TITAN_TEST(Trainer, GradAccumEquivalence) {
    // Two steps with (B=4, accum=1) and with (B=2, accum=2) see the same four
    // windows per step (sequential loaders), so the averaged gradient and
    // therefore the updated parameters agree.
    const auto run_one_step = [](std::int64_t batch, std::int64_t accum) {
        Model m(llama_style(5, 16, 1, 2, 1, 8));
        Rng init(21);
        m.init(init);
        TrainConfig c = small_config();
        c.batch = batch;
        c.grad_accum = accum;
        c.steps = 2;
        c.warmup = 1;
        c.eval_interval = 100;
        c.log_interval = 100;
        std::ostringstream log;
        Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100),
                        log);
        trainer.train_loader().set_sequential(true);
        trainer.run();
        std::vector<float> flat;
        for (const titan::nn::Parameter* p : m.parameters()) {
            const std::vector<float> v = p->value.to_vector();
            flat.insert(flat.end(), v.begin(), v.end());
        }
        return flat;
    };
    const std::vector<float> a = run_one_step(4, 1);
    const std::vector<float> b = run_one_step(2, 2);
    TITAN_EXPECT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        TITAN_EXPECT_NEAR(a[i], b[i], 1e-5);
    }
}
