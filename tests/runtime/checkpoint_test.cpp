#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "io/checkpoint_file.hpp"
#include "io/json.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "nn/module.hpp"
#include "optim/adamw.hpp"
#include "runtime/checkpoint.hpp"
#include "runtime/trainer.hpp"
#include "support/parity.hpp"

namespace {

using titan::Device;
using titan::Rng;
using titan::Tensor;
using titan::io::JsonValue;
using titan::model::llama_style;
using titan::model::Model;
using titan::runtime::Checkpoint;
using titan::runtime::load_checkpoint;
using titan::runtime::save_checkpoint;

std::filesystem::path temp_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

titan::runtime::TrainConfig sample_train_config() {
    return titan::runtime::parse_train_config(titan::io::parse_json(R"({
        "batch": 2, "seq": 4, "steps": 10, "lr": 1e-3, "min_lr": 1e-4,
        "warmup": 2, "eval_interval": 5, "eval_batches": 1,
        "log_interval": 5, "seed": 7, "checkpoint_interval": 3
    })"));
}

// A checkpoint of `model` whose optimizer moments come from two AdamW steps
// on made-up gradients, so they are neither zero nor equal to each other.
Checkpoint sample_checkpoint(Model& model) {
    titan::optim::AdamW opt(model.parameters(), titan::optim::AdamWConfig{});
    Rng rng(5);
    for (int step = 0; step < 2; ++step) {
        for (titan::nn::Parameter* p : model.parameters()) {
            Tensor g = Tensor::empty(p->value.shape(), Device::CPU);
            rng.fill_normal(g, 0.0f, 1.0f);
            p->grad = g.to(p->value.device());
        }
        opt.step(1e-3f);
    }
    Checkpoint c;
    c.model = model.config();
    c.train = sample_train_config();
    c.tokenizer =
        titan::io::parse_json(R"({"type": "char", "vocab": "abcde"})");
    c.state.step = 2;
    c.state.optimizer = opt.state();
    Rng train_rng(1), val_rng(2);
    train_rng.normal();  // leaves a cached normal in the saved state
    c.state.train_rng = train_rng.save_state();
    c.state.val_rng = val_rng.save_state();
    c.state.train_tokens = 400;
    c.state.val_tokens = 100;
    // The largest CRC-32 checks that the full 32-bit range round-trips.
    c.state.train_crc32 = 0xFFFFFFFFu;
    c.state.val_crc32 = 12345u;
    return c;
}

const std::vector<std::int32_t> kIds = {0, 1, 2, 3, 4, 3, 2, 1};

Tensor logits_of(Model& m) {
    titan::nn::ForwardContext ctx;
    ctx.training = false;
    return m.forward(kIds, 2, 4, ctx);
}

std::vector<float> flat(const std::vector<Tensor>& ts) {
    std::vector<float> out;
    for (const Tensor& t : ts) {
        const std::vector<float> v = t.to_vector();
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

// The message of the titan::Error thrown by load_checkpoint, or "".
std::string load_error(const std::filesystem::path& p, Model* m) {
    try {
        load_checkpoint(p, m);
    } catch (const titan::Error& e) {
        return e.what();
    }
    return "";
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

// Rewrites the checkpoint at `p` with its tensors passed through `edit`.
template <typename Edit>
void rewrite_tensors(const std::filesystem::path& p, Edit edit) {
    titan::io::CheckpointFile f = titan::io::read_checkpoint_file(p);
    edit(f.tensors);
    titan::io::write_checkpoint_file(p, f.meta, f.tensors);
}

}  // namespace

TITAN_TEST(RuntimeCheckpoint, RoundTripRestoresModel) {
    Model a(llama_style(5, 16, 2, 2, 1, 8));
    Rng init(3);
    a.init(init);
    const Checkpoint saved = sample_checkpoint(a);
    const Tensor before = logits_of(a);
    const auto p = temp_path("titan_rt_ckpt_roundtrip.tckp");
    save_checkpoint(p, a, saved);

    // The header alone gives everything but the moments.
    const Checkpoint header = load_checkpoint(p, nullptr);
    TITAN_EXPECT(header.model == saved.model);
    TITAN_EXPECT(header.train == saved.train);
    TITAN_EXPECT(header.state.optimizer.m.empty());

    // A model built from the header, with different weights, gets a's back.
    Model b(header.model);
    Rng other(99);
    b.init(other);
    const Checkpoint loaded = load_checkpoint(p, &b);
    const std::vector<float> want = before.to_vector();
    const std::vector<float> got = logits_of(b).to_vector();
    TITAN_EXPECT(want == got);

    TITAN_EXPECT(loaded.model == saved.model);
    TITAN_EXPECT(loaded.train == saved.train);
    TITAN_EXPECT(titan::io::to_json(loaded.tokenizer) ==
                 titan::io::to_json(saved.tokenizer));
    TITAN_EXPECT_EQ(loaded.state.step, saved.state.step);
    TITAN_EXPECT_EQ(loaded.state.optimizer.steps, saved.state.optimizer.steps);
    TITAN_EXPECT(loaded.state.train_rng == saved.state.train_rng);
    TITAN_EXPECT(loaded.state.val_rng == saved.state.val_rng);
    TITAN_EXPECT_EQ(loaded.state.train_tokens, saved.state.train_tokens);
    TITAN_EXPECT_EQ(loaded.state.val_tokens, saved.state.val_tokens);
    TITAN_EXPECT_EQ(loaded.state.train_crc32, saved.state.train_crc32);
    TITAN_EXPECT_EQ(loaded.state.val_crc32, saved.state.val_crc32);
    TITAN_EXPECT(flat(loaded.state.optimizer.m) ==
                 flat(saved.state.optimizer.m));
    TITAN_EXPECT(flat(loaded.state.optimizer.v) ==
                 flat(saved.state.optimizer.v));
    TITAN_EXPECT(flat(loaded.state.optimizer.m) !=
                 flat(loaded.state.optimizer.v));
    std::filesystem::remove(p);
}

TITAN_TEST(RuntimeCheckpoint, CudaSaveCpuLoad) {
    TITAN_REQUIRE_CUDA();
    Model gpu(llama_style(5, 16, 2, 2, 1, 8));
    Rng init(4);
    gpu.init(init);
    gpu.to(Device::CUDA);
    const Checkpoint saved = sample_checkpoint(gpu);
    const Tensor gpu_logits = logits_of(gpu);
    const auto p = temp_path("titan_rt_ckpt_cuda.tckp");
    save_checkpoint(p, gpu, saved);

    Model cpu(llama_style(5, 16, 2, 2, 1, 8));
    const Checkpoint loaded = load_checkpoint(p, &cpu);
    TITAN_EXPECT(cpu.device() == Device::CPU);
    TITAN_EXPECT_PARITY(logits_of(cpu), gpu_logits, titan::test::kParityAtol,
                        titan::test::kParityRtol);
    TITAN_EXPECT(flat(loaded.state.optimizer.m) ==
                 flat(saved.state.optimizer.m));
    std::filesystem::remove(p);
}

TITAN_TEST(RuntimeCheckpoint, RejectsShapeMismatch) {
    Model wide(llama_style(5, 64, 1, 2, 1, 8));
    const auto p = temp_path("titan_rt_ckpt_shape.tckp");
    save_checkpoint(p, wide, sample_checkpoint(wide));

    Model narrow(llama_style(5, 32, 1, 2, 1, 8));
    Rng init(6);
    narrow.init(init);
    const std::vector<float> before = logits_of(narrow).to_vector();
    const std::string message = load_error(p, &narrow);
    TITAN_EXPECT(contains(message, p.string()));
    TITAN_EXPECT(contains(message, "tensor embed_tokens.weight"));
    TITAN_EXPECT(contains(message, "[5, 64] in the checkpoint"));
    TITAN_EXPECT(contains(message, "[5, 32] in the model"));
    // The model is left as it was.
    TITAN_EXPECT(logits_of(narrow).to_vector() == before);
    std::filesystem::remove(p);
}

TITAN_TEST(RuntimeCheckpoint, RejectsConfigMismatch) {
    // Same tensor shapes, different RoPE base: only the config check sees it.
    titan::model::ModelConfig saved_config = llama_style(5, 16, 1, 2, 1, 8);
    Model saved(saved_config);
    const auto p = temp_path("titan_rt_ckpt_config.tckp");
    save_checkpoint(p, saved, sample_checkpoint(saved));

    titan::model::ModelConfig other = saved_config;
    other.rope_theta = 500000.0f;
    Model target(other);
    Rng init(8);
    target.init(init);
    const std::vector<float> before = logits_of(target).to_vector();
    const std::string message = load_error(p, &target);
    TITAN_EXPECT(contains(message, p.string()));
    TITAN_EXPECT(contains(message,
                          "model.rope_theta is 10000 in the "
                          "checkpoint but 500000 in the model"));
    TITAN_EXPECT(logits_of(target).to_vector() == before);
    std::filesystem::remove(p);
}

TITAN_TEST(RuntimeCheckpoint, TokenizerMustMatchVocab) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    Checkpoint c = sample_checkpoint(m);
    c.tokenizer = titan::io::parse_json(
        R"({"type": "char", "vocab": ["a", "b", "c", "d", "e"]})");
    const std::filesystem::path p = "run/latest.ckpt";
    // Five tokens, as the model expects.
    TITAN_EXPECT_EQ(titan::runtime::checkpoint_tokenizer(p, c).vocab_size(),
                    std::int64_t{5});

    const auto error = [&](const char* tokenizer_json) {
        Checkpoint bad = c;
        bad.tokenizer = titan::io::parse_json(tokenizer_json);
        try {
            titan::runtime::checkpoint_tokenizer(p, bad);
        } catch (const titan::Error& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::string fewer =
        error(R"({"type": "char", "vocab": ["a", "b", "c"]})");
    TITAN_EXPECT(contains(fewer, p.string()));
    TITAN_EXPECT(contains(fewer,
                          "the tokenizer has 3 tokens but "
                          "meta.vocab_size is 5"));
    const std::string malformed = error(R"({"type": "bpe"})");
    TITAN_EXPECT(contains(malformed, p.string()));
    TITAN_EXPECT(contains(malformed, "meta.tokenizer"));
}

TITAN_TEST(RuntimeCheckpoint, RejectsMissingTensor) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    const auto p = temp_path("titan_rt_ckpt_missing.tckp");
    for (const std::string gone :
         {"final_norm.weight", "optim.v.final_norm.weight"}) {
        save_checkpoint(p, m, sample_checkpoint(m));
        rewrite_tensors(p, [&](std::vector<titan::io::NamedTensor>& ts) {
            std::erase_if(ts, [&](const titan::io::NamedTensor& t) {
                return t.name == gone;
            });
        });
        const std::string message = load_error(p, &m);
        TITAN_EXPECT(contains(message, p.string()));
        TITAN_EXPECT(contains(message, "missing tensor " + gone));
    }
    std::filesystem::remove(p);
}

TITAN_TEST(RuntimeCheckpoint, RejectsExtraTensor) {
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    const auto p = temp_path("titan_rt_ckpt_extra.tckp");
    save_checkpoint(p, m, sample_checkpoint(m));
    rewrite_tensors(p, [](std::vector<titan::io::NamedTensor>& ts) {
        ts.push_back({"lm_head.weight", Tensor::zeros({5, 16}, Device::CPU)});
    });
    const std::string message = load_error(p, &m);
    TITAN_EXPECT(contains(message, p.string()));
    TITAN_EXPECT(contains(message, "unexpected tensor lm_head.weight"));
    std::filesystem::remove(p);
}

TITAN_TEST(RuntimeCheckpoint, RejectsMalformedMeta) {
    // Each case replaces one part of a valid meta; the error names its path.
    Model m(llama_style(5, 16, 1, 2, 1, 8));
    const auto p = temp_path("titan_rt_ckpt_meta.tckp");
    save_checkpoint(p, m, sample_checkpoint(m));
    const titan::io::CheckpointFile good = titan::io::read_checkpoint_file(p);

    struct Case {
        const char* key;
        JsonValue value;
        const char* needle;
    };
    JsonValue bad_state = good.meta.at("state");
    bad_state.set("train_rng", JsonValue::string("not a state"));
    JsonValue negative_step = good.meta.at("state");
    negative_step.set("step", JsonValue::number(-1));
    JsonValue bad_train = good.meta.at("train");
    bad_train.set("lr", JsonValue::number(0));
    JsonValue bad_model = good.meta.at("model");
    bad_model.set("dim", JsonValue::string("16"));
    JsonValue big_crc = good.meta.at("state");
    big_crc.set("train_crc32", JsonValue::number(4294967296.0));
    JsonValue state_typo = good.meta.at("state");
    state_typo.set("stpe", JsonValue::number(1));
    const std::vector<Case> cases = {
        {"state", bad_state, "meta.state.train_rng"},
        {"state", negative_step, "meta.state.step"},
        {"state", state_typo, "meta.state.stpe"},
        {"state", big_crc, "meta.state.train_crc32"},
        {"train", bad_train, "train.lr"},
        {"model", bad_model, "model.dim"},
        {"vocab_size", JsonValue::number(0), "meta.vocab_size"},
        {"extra", JsonValue::number(1), "meta.extra"},
    };
    for (const Case& c : cases) {
        JsonValue meta = good.meta;
        meta.set(c.key, c.value);
        titan::io::write_checkpoint_file(p, meta, good.tensors);
        for (Model* target : {static_cast<Model*>(nullptr), &m}) {
            const std::string message = load_error(p, target);
            TITAN_EXPECT(contains(message, p.string()));
            TITAN_EXPECT(contains(message, c.needle));
        }
    }
    std::filesystem::remove(p);
}
