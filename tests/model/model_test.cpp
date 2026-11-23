#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "nn/init.hpp"
#include "nn/loss.hpp"
#include "optim/adamw.hpp"
#include "support/gradcheck.hpp"

using titan::Rng;
using titan::Tensor;
using titan::model::llama_style;
using titan::model::Model;
using titan::model::ModelConfig;
using titan::nn::CrossEntropyLoss;
using titan::nn::ForwardContext;
using titan::nn::Parameter;
using titan::test::check_gradient;

namespace {

std::vector<std::int32_t> random_ids(std::size_t n, std::int64_t vocab,
                                     Rng& rng) {
    std::vector<std::int32_t> ids(n);
    for (std::int32_t& id : ids) {
        id = static_cast<std::int32_t>(rng.randint(vocab));
    }
    return ids;
}

Parameter& find_parameter(Model& m, const std::string& name) {
    for (Parameter* p : m.parameters()) {
        if (p->name == name) {
            return *p;
        }
    }
    throw titan::Error("no parameter named " + name);
}

}  // namespace

TITAN_TEST(Model, ParameterCountTiny) {
    // vocab 65, dim 16, 2 layers, 2 heads (hd 8), 1 KV head, tied embeddings.
    // SwiGLU hidden = round_up(8·16/3, 8) = round_up(42, 8) = 48.
    Model m(llama_style(65, 16, 2, 2, 1, 32));
    const std::int64_t embed = 65 * 16;    // embed_tokens, shared with lm_head
    const std::int64_t block = 16          // norm1
                               + 16 * 16   // q_proj: 2 heads · 8
                               + 8 * 16    // k_proj: 1 KV head · 8
                               + 8 * 16    // v_proj
                               + 16 * 16   // o_proj
                               + 16        // norm2
                               + 48 * 16   // gate_proj
                               + 48 * 16   // up_proj
                               + 16 * 48;  // down_proj
    const std::int64_t final_norm = 16;
    TITAN_EXPECT_EQ(m.num_parameters(), embed + 2 * block + final_norm);
    TITAN_EXPECT_EQ(m.num_parameters(), std::int64_t{7264});
}

TITAN_TEST(Model, TiedParameterListedOnce) {
    ModelConfig c = llama_style(11, 8, 2, 2, 1, 8);
    Model tied(c);
    const std::vector<Parameter*> params = tied.parameters();
    std::set<const Parameter*> unique_ptrs(params.begin(), params.end());
    std::set<std::string> unique_names;
    for (const Parameter* p : params) {
        unique_names.insert(p->name);
    }
    TITAN_EXPECT_EQ(unique_ptrs.size(), params.size());
    TITAN_EXPECT_EQ(unique_names.size(), params.size());
    TITAN_EXPECT(unique_names.contains("embed_tokens.weight"));
    TITAN_EXPECT(unique_names.contains("blocks.1.attn.q_proj.weight"));
    TITAN_EXPECT(unique_names.contains("blocks.0.norm1.weight"));
    TITAN_EXPECT(unique_names.contains("final_norm.weight"));
    TITAN_EXPECT(!unique_names.contains("lm_head.weight"));

    c.tied_embeddings = false;
    Model untied(c);
    bool has_head = false;
    for (const Parameter* p : untied.parameters()) {
        has_head = has_head || p->name == "lm_head.weight";
    }
    TITAN_EXPECT(has_head);
    TITAN_EXPECT_EQ(untied.num_parameters(), tied.num_parameters() + 11 * 8);
}

namespace {

// Root mean square of a parameter's values: its stddev, the mean being 0.
double stddev(const Parameter& p) {
    double sq = 0.0;
    for (float v : p.value.to_vector()) sq += static_cast<double>(v) * v;
    return std::sqrt(sq / static_cast<double>(p.value.numel()));
}

}  // namespace

TITAN_TEST(Model, InitFollowsTheScheme) {
    // 4 layers: output projections use 0.02 / sqrt(2·4) ≈ 0.00707.
    Model m(llama_style(50, 32, 4, 4, 2, 16));
    Rng rng(60);
    m.init(rng);
    TITAN_EXPECT_NEAR(stddev(find_parameter(m, "embed_tokens.weight")), 0.02,
                      0.002);
    TITAN_EXPECT_NEAR(stddev(find_parameter(m, "blocks.0.attn.q_proj.weight")),
                      0.02, 0.002);
    TITAN_EXPECT_NEAR(stddev(find_parameter(m, "blocks.2.attn.o_proj.weight")),
                      0.00707, 0.0007);
    TITAN_EXPECT_NEAR(
        stddev(find_parameter(m, "blocks.3.mlp.down_proj.weight")), 0.00707,
        0.0007);
    for (float v : find_parameter(m, "final_norm.weight").value.to_vector()) {
        TITAN_EXPECT_EQ(v, 1.0f);
    }
}

TITAN_TEST(Model, InitFollowsTheSchemeGptStyle) {
    // LayerNorm + GELU blocks: the MLP's proj is the residual projection,
    // and the norm and linear biases start at 0.
    ModelConfig c = llama_style(50, 32, 4, 4, 2, 16);
    for (titan::model::BlockConfig& b : c.blocks) {
        b.mlp = "gelu";
        b.mlp_hidden = 64;
        b.norm = "layernorm";
        b.position = "none";
    }
    c.final_norm = "layernorm";
    Model m(c);
    // Non-zero values everywhere first, so init() must set every one.
    for (Parameter* p : m.parameters()) {
        titan::nn::init_constant(*p, 0.3f);
    }
    Rng rng(63);
    m.init(rng);
    TITAN_EXPECT_NEAR(stddev(find_parameter(m, "blocks.1.mlp.fc.weight")), 0.02,
                      0.002);
    TITAN_EXPECT_NEAR(stddev(find_parameter(m, "blocks.1.mlp.proj.weight")),
                      0.00707, 0.0007);
    for (const char* name : {"blocks.1.mlp.fc.bias", "blocks.1.mlp.proj.bias",
                             "blocks.1.norm1.bias", "final_norm.bias"}) {
        for (float v : find_parameter(m, name).value.to_vector()) {
            TITAN_EXPECT_EQ(v, 0.0f);
        }
    }
    for (const char* name : {"blocks.1.norm2.weight", "final_norm.weight"}) {
        for (float v : find_parameter(m, name).value.to_vector()) {
            TITAN_EXPECT_EQ(v, 1.0f);
        }
    }
}

TITAN_TEST(Model, GradCheckTiny) {
    Rng rng(61);
    const std::int64_t batch = 2;
    const std::int64_t t_len = 6;
    const std::int64_t vocab = 11;
    Model m(llama_style(vocab, 8, 2, 2, 1, 8));
    // Larger weights than init() gives, so that no gradient is close to zero.
    for (Parameter* p : m.parameters()) {
        titan::nn::init_normal(*p, rng, 0.5f);
    }
    const std::vector<std::int32_t> ids = random_ids(12, vocab, rng);
    const std::vector<std::int32_t> targets = random_ids(12, vocab, rng);

    // The scalar loss as a {1} tensor, so that check_gradient with r = 1
    // differentiates the loss itself.
    const auto loss_of = [&]() {
        ForwardContext ctx{.training = false, .pos_offset = 0};
        CrossEntropyLoss loss;
        const float value =
            loss.forward(m.forward(ids, batch, t_len, ctx), targets);
        return Tensor::from_vector(std::vector<float>{value}, {1});
    };
    const Tensor r = Tensor::from_vector(std::vector<float>{1.0f}, {1});

    titan::nn::zero_grad(m.parameters());
    ForwardContext ctx;
    CrossEntropyLoss loss;
    loss.forward(m.forward(ids, batch, t_len, ctx), targets);
    m.backward(loss.backward());

    for (const char* name :
         {"embed_tokens.weight", "blocks.0.attn.q_proj.weight",
          "blocks.0.mlp.down_proj.weight"}) {
        Parameter& p = find_parameter(m, name);
        const Tensor analytic = p.grad.clone();
        const titan::test::GradCheckResult result =
            check_gradient(loss_of, p.value, analytic, r);
        if (!result.ok) {
            titan::test::report_failure(
                __FILE__, __LINE__, std::string(name) + ": " + result.detail);
        }
    }
}

TITAN_TEST(Model, OverfitsSingleBatch) {
    // NFR-3: a small model memorizes one fixed batch.
    const std::int64_t batch = 4;
    const std::int64_t t_len = 16;
    const std::int64_t vocab = 20;
    Model m(llama_style(vocab, 32, 2, 4, 2, 16));
    Rng rng(62);
    m.init(rng);
    // One random sequence of t_len + 1 tokens per row; the targets are the
    // inputs shifted left by one.
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> targets;
    for (std::int64_t b = 0; b < batch; ++b) {
        const std::vector<std::int32_t> seq =
            random_ids(static_cast<std::size_t>(t_len + 1), vocab, rng);
        ids.insert(ids.end(), seq.begin(), seq.end() - 1);
        targets.insert(targets.end(), seq.begin() + 1, seq.end());
    }

    titan::optim::AdamWConfig cfg;
    titan::optim::AdamW opt(m.parameters(), cfg);
    CrossEntropyLoss loss;
    float value = 0.0f;
    for (int step = 0; step < 300; ++step) {
        titan::nn::zero_grad(m.parameters());
        ForwardContext ctx;
        value = loss.forward(m.forward(ids, batch, t_len, ctx), targets);
        if (value < 0.05f) {
            break;
        }
        m.backward(loss.backward());
        opt.step(3e-3f);
    }
    TITAN_EXPECT(value < 0.05f);
}

TITAN_TEST(Model, SequenceLongerThanContextThrows) {
    Model m(llama_style(11, 8, 1, 2, 1, 4));
    ForwardContext ctx;
    const std::vector<std::int32_t> ids(5, 1);
    TITAN_EXPECT_THROWS(m.forward(ids, 1, 5, ctx), titan::Error);
    // The id count must match B·T.
    TITAN_EXPECT_THROWS(m.forward(ids, 1, 4, ctx), titan::Error);
    const std::vector<std::int32_t> four(4, 1);
    const Tensor logits = m.forward(four, 1, 4, ctx);
    TITAN_EXPECT(logits.shape() == titan::Shape({1, 4, 11}));
}

TITAN_TEST(Model, UnknownVariantStringsThrow) {
    ModelConfig c = llama_style(11, 8, 1, 2, 1, 4);
    c.blocks[0].mlp = "relu";
    TITAN_EXPECT_THROWS(Model{c}, titan::Error);
    c = llama_style(11, 8, 1, 2, 1, 4);
    c.blocks[0].position = "alibi";
    TITAN_EXPECT_THROWS(Model{c}, titan::Error);
    c = llama_style(11, 8, 1, 2, 1, 4);
    c.final_norm = "batchnorm";
    TITAN_EXPECT_THROWS(Model{c}, titan::Error);
}
