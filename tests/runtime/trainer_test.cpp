#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/rng.hpp"
#include "harness/test.hpp"
#include "io/json.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "runtime/checkpoint.hpp"
#include "runtime/trainer.hpp"
#include "support/parity.hpp"

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

TITAN_TEST(Trainer, RejectsZeroSizes) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    std::ostringstream log;
    for (int which = 0; which < 4; ++which) {
        TrainConfig c = small_config();
        (which == 0   ? c.grad_accum
         : which == 1 ? c.log_interval
         : which == 2 ? c.eval_interval
                      : c.eval_batches) = 0;
        TITAN_EXPECT_THROWS(
            Trainer(m, c, repeating_tokens(400), repeating_tokens(100), log),
            titan::Error);
    }
}

TITAN_TEST(Trainer, GradAccumEquivalence) {
    // (B=4, accum=1) and (B=2, accum=2) see the same four windows per step
    // (sequential loaders). Their gradients must match: the accumulated
    // micro-batch gradients carry the 1/grad_accum scale. Adam is
    // scale-invariant and clipping normalises, so with a huge grad_clip the
    // gradients left in the parameters by the last step are the only direct
    // witness of that scale; the updated parameters are compared as well.
    struct Outcome {
        std::vector<float> values, grads;
    };
    const auto run_two_steps = [](std::int64_t batch, std::int64_t accum) {
        Model m(llama_style(5, 16, 1, 2, 1, 8));
        Rng init(21);
        m.init(init);
        TrainConfig c = small_config();
        c.batch = batch;
        c.grad_accum = accum;
        c.grad_clip = 1e30f;
        c.steps = 2;
        c.warmup = 1;
        c.eval_interval = 100;
        c.log_interval = 100;
        std::ostringstream log;
        Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100),
                        log);
        trainer.train_loader().set_sequential(true);
        trainer.run();
        Outcome out;
        for (const titan::nn::Parameter* p : m.parameters()) {
            const std::vector<float> v = p->value.to_vector();
            const std::vector<float> g = p->grad.to_vector();
            out.values.insert(out.values.end(), v.begin(), v.end());
            out.grads.insert(out.grads.end(), g.begin(), g.end());
        }
        return out;
    };
    const Outcome a = run_two_steps(4, 1);
    const Outcome b = run_two_steps(2, 2);
    TITAN_EXPECT_EQ(a.values.size(), b.values.size());
    double max_grad = 0.0;
    for (std::size_t i = 0; i < a.values.size(); ++i) {
        TITAN_EXPECT_NEAR(a.values[i], b.values[i], 1e-5);
        TITAN_EXPECT_NEAR(a.grads[i], b.grads[i], 1e-5);
        max_grad =
            std::max(max_grad, std::fabs(static_cast<double>(a.grads[i])));
    }
    // The comparison is only meaningful if the gradients are not all ~0.
    TITAN_EXPECT(max_grad > 1e-3);
}

namespace {

struct SeededOutcome {
    titan::runtime::TrainResult result;
    std::vector<float> values;
};

// A short seeded run on `device`; returns the result and every parameter.
SeededOutcome seeded_run(titan::Device device) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    Rng init(31);
    m.init(init);
    TrainConfig c = small_config();
    c.steps = 20;
    c.warmup = 2;
    c.eval_interval = 10;
    c.log_interval = 10;
    c.device = device;
    std::ostringstream log;
    Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100), log);
    TITAN_EXPECT(m.device() == device);
    SeededOutcome out;
    out.result = trainer.run();
    for (const titan::nn::Parameter* p : m.parameters()) {
        const std::vector<float> v = p->value.to_vector();
        out.values.insert(out.values.end(), v.begin(), v.end());
    }
    return out;
}

// Records a failure unless the two runs agree bit for bit.
void expect_identical(const SeededOutcome& a, const SeededOutcome& b) {
    TITAN_EXPECT(a.result.final_train_loss == b.result.final_train_loss);
    TITAN_EXPECT(a.result.final_val_loss == b.result.final_val_loss);
    TITAN_EXPECT_EQ(a.values.size(), b.values.size());
    bool identical = a.values.size() == b.values.size();
    for (std::size_t i = 0; identical && i < a.values.size(); ++i) {
        identical = a.values[i] == b.values[i];
    }
    TITAN_EXPECT(identical);
}

}  // namespace

TITAN_TEST(Trainer, SameSeedIsBitReproducible) {
    // The same seed and config must give the same run, bit for bit: init,
    // data order and every optimizer step are deterministic.
    expect_identical(seeded_run(titan::Device::CPU),
                     seeded_run(titan::Device::CPU));
}

TITAN_TEST(Trainer, CudaSameSeedIsBitReproducible) {
    // The same on the GPU: the kernels use fixed reduction orders and no
    // float atomics, so two runs on the same GPU agree bit for bit.
    TITAN_REQUIRE_CUDA();
    expect_identical(seeded_run(titan::Device::CUDA),
                     seeded_run(titan::Device::CUDA));
}

namespace {

// A config for the checkpoint tests: steps 12, evaluation every 4 steps.
TrainConfig resume_config(titan::Device device) {
    TrainConfig c = small_config();
    c.steps = 12;
    c.warmup = 2;
    c.eval_interval = 4;
    c.log_interval = 4;
    c.checkpoint_interval = 4;
    c.device = device;
    return c;
}

std::vector<float> all_values(Model& m) {
    std::vector<float> out;
    for (const titan::nn::Parameter* p : m.parameters()) {
        const std::vector<float> v = p->value.to_vector();
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

// Run A trains 12 steps without a break. Run B stops after step 6, saves a
// checkpoint from on_checkpoint, and a fresh model and trainer built from it
// train to 12. Both must agree bit for bit.
void expect_resume_identical(titan::Device device, const char* file) {
    const TrainConfig c = resume_config(device);
    std::ostringstream log;

    SeededOutcome a;
    {
        Model m(llama_style(5, 16, 1, 2, 1, 8));
        Rng init(31);
        m.init(init);
        Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100),
                        log);
        a.result = trainer.run();
        a.values = all_values(m);
    }

    const auto path = std::filesystem::temp_directory_path() / file;
    {
        Model m(llama_style(5, 16, 1, 2, 1, 8));
        Rng init(31);
        m.init(init);
        Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100),
                        log);
        trainer.on_checkpoint([&](std::int64_t) {
            titan::runtime::Checkpoint ckpt;
            ckpt.model = m.config();
            ckpt.train = c;
            ckpt.tokenizer = titan::io::JsonValue::object();
            ckpt.state = trainer.state();
            titan::runtime::save_checkpoint(path, m, ckpt);
        });
        trainer.run(6);
        TITAN_EXPECT_EQ(trainer.state().step, std::int64_t{6});
    }

    // The resume path of `titan train --resume`: the header gives the model,
    // the full load fills it, then the trainer moves it to the device.
    const titan::runtime::Checkpoint header =
        titan::runtime::load_checkpoint(path, nullptr);
    Model m(header.model);
    const titan::runtime::Checkpoint ckpt =
        titan::runtime::load_checkpoint(path, &m);
    TITAN_EXPECT_EQ(ckpt.state.step, std::int64_t{6});
    TrainConfig resumed = ckpt.train;
    resumed.device = device;
    Trainer trainer(m, resumed, repeating_tokens(400), repeating_tokens(100),
                    log, ckpt.state);
    SeededOutcome b;
    b.result = trainer.run();
    b.values = all_values(m);
    // Only steps 7..12 ran in the resumed trainer.
    TITAN_EXPECT_EQ(b.result.tokens, std::int64_t{6 * 4 * 8});
    expect_identical(a, b);
    std::filesystem::remove(path);
}

}  // namespace

TITAN_TEST(Trainer, ResumeIsBitIdentical) {
    expect_resume_identical(titan::Device::CPU, "titan_resume_cpu.tckp");
    if (titan::cuda_available()) {
        expect_resume_identical(titan::Device::CUDA, "titan_resume_cuda.tckp");
    }
}

TITAN_TEST(Trainer, ResumeRejectsChangedData) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    Rng init(11);
    m.init(init);
    const TrainConfig c = resume_config(titan::Device::CPU);
    std::ostringstream log;
    titan::runtime::TrainerState state;
    {
        Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100),
                        log);
        trainer.run(2);
        state = trainer.state();
    }
    TITAN_EXPECT_EQ(state.train_tokens, std::int64_t{400});
    TITAN_EXPECT_EQ(state.val_tokens, std::int64_t{100});

    const auto resume_error = [&](std::size_t train_n, std::size_t val_n) {
        try {
            Trainer trainer(m, c, repeating_tokens(train_n),
                            repeating_tokens(val_n), log, state);
        } catch (const titan::Error& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    TITAN_EXPECT(contains(resume_error(401, 100),
                          "resume: train tokens changed (checkpoint 400, "
                          "now 401)"));
    TITAN_EXPECT(contains(resume_error(400, 99),
                          "resume: val tokens changed (checkpoint 100, "
                          "now 99)"));
    TITAN_EXPECT(resume_error(400, 100).empty());

    // Same lengths, different content: only the checksums tell.
    std::vector<std::int32_t> edited = repeating_tokens(400);
    edited[17] = (edited[17] + 1) % 5;
    const auto content_error = [&](std::vector<std::int32_t> train,
                                   std::vector<std::int32_t> val) {
        try {
            Trainer trainer(m, c, std::move(train), std::move(val), log, state);
        } catch (const titan::Error& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    TITAN_EXPECT(contains(content_error(edited, repeating_tokens(100)),
                          "resume: train tokens changed (checkpoint crc32 "));
    std::vector<std::int32_t> edited_val = repeating_tokens(100);
    edited_val[0] = (edited_val[0] + 1) % 5;
    TITAN_EXPECT(contains(content_error(repeating_tokens(400), edited_val),
                          "resume: val tokens changed (checkpoint crc32 "));
    TITAN_EXPECT_EQ(state.train_crc32,
                    titan::runtime::token_crc32(repeating_tokens(400)));
}

TITAN_TEST(Trainer, CheckpointCallbackCadence) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    Rng init(11);
    m.init(init);
    TrainConfig c = small_config();
    c.steps = 10;
    c.checkpoint_interval = 4;
    c.warmup = 2;
    std::ostringstream log;
    Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100), log);
    std::vector<std::int64_t> calls;
    trainer.on_checkpoint([&](std::int64_t step) {
        calls.push_back(step);
        // The callback sees the state after that step.
        TITAN_EXPECT_EQ(trainer.state().step, step);
    });
    trainer.run();
    TITAN_EXPECT(calls == (std::vector<std::int64_t>{4, 8, 10}));
}

TITAN_TEST(Trainer, StopAfterFiresCheckpoint) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    Rng init(11);
    m.init(init);
    TrainConfig c = small_config();
    c.steps = 10;
    c.checkpoint_interval = 4;
    c.warmup = 2;
    std::ostringstream log;
    Trainer trainer(m, c, repeating_tokens(400), repeating_tokens(100), log);
    std::vector<std::int64_t> calls;
    trainer.on_checkpoint([&](std::int64_t step) { calls.push_back(step); });
    TITAN_EXPECT_THROWS(trainer.run(0), titan::Error);
    TITAN_EXPECT_THROWS(trainer.run(11), titan::Error);

    // 6 is not a multiple of 4, yet a stopped run always gets a checkpoint.
    const titan::runtime::TrainResult r = trainer.run(6);
    TITAN_EXPECT(calls == (std::vector<std::int64_t>{4, 6}));
    TITAN_EXPECT_EQ(trainer.state().step, std::int64_t{6});
    TITAN_EXPECT_EQ(r.tokens, std::int64_t{6 * 4 * 8});
    // Steps already run cannot be a stop point; the rest of the run can.
    TITAN_EXPECT_THROWS(trainer.run(6), titan::Error);
    trainer.run();
    TITAN_EXPECT(calls == (std::vector<std::int64_t>{4, 6, 8, 10}));
    // A finished run has nothing left to run.
    TITAN_EXPECT_EQ(trainer.run().tokens, std::int64_t{0});
    TITAN_EXPECT_EQ(calls.size(), std::size_t{4});
}
