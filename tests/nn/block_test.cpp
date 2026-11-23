#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/attention.hpp"
#include "nn/block.hpp"
#include "nn/init.hpp"
#include "nn/lm_head.hpp"
#include "nn/loss.hpp"
#include "ops/cross_entropy.hpp"
#include "support/gradcheck.hpp"
#include "support/module_gradcheck.hpp"

using titan::Rng;
using titan::Tensor;
using titan::nn::BlockSpec;
using titan::nn::CrossEntropyLoss;
using titan::nn::ForwardContext;
using titan::nn::LMHead;
using titan::nn::MlpType;
using titan::nn::NormType;
using titan::nn::Parameter;
using titan::nn::PositionEncoding;
using titan::nn::TransformerBlock;
using titan::test::check_gradient;
using titan::test::check_module_gradients;
using titan::test::random_tensor;
using titan::test::randomize_parameters;

namespace {

constexpr float kEps = 1e-5f;
constexpr float kTheta = 10000.0f;

std::vector<std::string> names_of(titan::nn::Module& m) {
    std::vector<Parameter*> params;
    m.parameters(params);
    std::vector<std::string> names;
    for (const Parameter* p : params) {
        names.push_back(p->name);
    }
    return names;
}

}  // namespace

TITAN_TEST(Block, GradCheckLlamaStyle) {
    Rng rng(50);
    const BlockSpec spec{.n_heads = 4,
                         .n_kv_heads = 2,
                         .mlp_hidden = 12,
                         .mlp = MlpType::SwiGLU,
                         .norm = NormType::RMSNorm,
                         .position = PositionEncoding::Rope};
    TransformerBlock block("block", 8, spec, kEps, kTheta);
    randomize_parameters(block, rng, 0.5f);
    Tensor x = random_tensor({2, 5, 8}, rng);
    check_module_gradients(block, x, rng);
}

TITAN_TEST(Block, GradCheckGptStyle) {
    Rng rng(51);
    const BlockSpec spec{.n_heads = 2,
                         .n_kv_heads = 2,
                         .mlp_hidden = 16,
                         .mlp = MlpType::Gelu,
                         .norm = NormType::LayerNorm,
                         .position = PositionEncoding::None};
    TransformerBlock block("block", 8, spec, kEps, kTheta);
    randomize_parameters(block, rng, 0.5f);
    Tensor x = random_tensor({2, 5, 8}, rng);
    check_module_gradients(block, x, rng);
}

TITAN_TEST(Block, ParameterNames) {
    const BlockSpec llama{.n_heads = 2,
                          .n_kv_heads = 1,
                          .mlp_hidden = 6,
                          .mlp = MlpType::SwiGLU,
                          .norm = NormType::RMSNorm,
                          .position = PositionEncoding::Rope};
    TransformerBlock a("blocks.0", 4, llama, kEps, kTheta);
    TITAN_EXPECT(
        names_of(a) ==
        std::vector<std::string>(
            {"blocks.0.norm1.weight", "blocks.0.attn.q_proj.weight",
             "blocks.0.attn.k_proj.weight", "blocks.0.attn.v_proj.weight",
             "blocks.0.attn.o_proj.weight", "blocks.0.norm2.weight",
             "blocks.0.mlp.gate_proj.weight", "blocks.0.mlp.up_proj.weight",
             "blocks.0.mlp.down_proj.weight"}));

    const BlockSpec gpt{.n_heads = 2,
                        .n_kv_heads = 2,
                        .mlp_hidden = 6,
                        .mlp = MlpType::Gelu,
                        .norm = NormType::LayerNorm,
                        .position = PositionEncoding::None};
    TransformerBlock b("blocks.1", 4, gpt, kEps, kTheta);
    TITAN_EXPECT(
        names_of(b) ==
        std::vector<std::string>(
            {"blocks.1.norm1.weight", "blocks.1.norm1.bias",
             "blocks.1.attn.q_proj.weight", "blocks.1.attn.k_proj.weight",
             "blocks.1.attn.v_proj.weight", "blocks.1.attn.o_proj.weight",
             "blocks.1.norm2.weight", "blocks.1.norm2.bias",
             "blocks.1.mlp.fc.weight", "blocks.1.mlp.fc.bias",
             "blocks.1.mlp.proj.weight", "blocks.1.mlp.proj.bias"}));
}

TITAN_TEST(LMHead, GradCheckUntied) {
    Rng rng(52);
    LMHead head("lm_head", 6, 9);
    randomize_parameters(head, rng, 0.5f);
    TITAN_EXPECT(names_of(head) ==
                 std::vector<std::string>({"lm_head.weight"}));
    Tensor x = random_tensor({2, 3, 6}, rng);
    check_module_gradients(head, x, rng);
}

TITAN_TEST(LMHead, TiedAccumulatesIntoSharedWeight) {
    Rng rng(53);
    Parameter e = titan::nn::make_parameter("embed_tokens.weight", {9, 6});
    titan::nn::init_normal(e, rng, 0.5f);
    LMHead head(e);
    TITAN_EXPECT(names_of(head).empty());

    Tensor x = random_tensor({2, 3, 6}, rng);
    const auto forward = [&]() {
        ForwardContext ctx;
        return head.forward(x, ctx);
    };
    const Tensor y = forward();
    TITAN_EXPECT(y.shape() == titan::Shape({2, 3, 9}));
    const Tensor r = random_tensor(y.shape(), rng);
    // The shared gradient already holds a value; backward must add to it.
    const std::vector<float> before(static_cast<std::size_t>(e.grad.numel()),
                                    0.25f);
    float* g = e.grad.data();
    for (std::size_t i = 0; i < before.size(); ++i) g[i] = before[i];
    const Tensor dx = head.backward(r);

    std::vector<float> accumulated = e.grad.to_vector();
    for (std::size_t i = 0; i < accumulated.size(); ++i) {
        accumulated[i] -= before[i];
    }
    const Tensor analytic = Tensor::from_vector(accumulated, e.value.shape());
    TITAN_EXPECT_GRAD_OK(check_gradient(forward, e.value, analytic, r));
    TITAN_EXPECT_GRAD_OK(check_gradient(forward, x, dx, r));
}

TITAN_TEST(CrossEntropyLoss, FlattensAndReturnsGradientInLogitsShape) {
    Rng rng(54);
    const Tensor logits = random_tensor({2, 3, 5}, rng);
    const std::vector<std::int32_t> targets = {0, 4, 2, 1, 3, 3};
    CrossEntropyLoss loss;
    const float value = loss.forward(logits, targets);

    Tensor expected_grad = Tensor::zeros({1, 1}, titan::Device::CPU);
    const float expected = titan::ops::cross_entropy(logits.reshape({6, 5}),
                                                     targets, &expected_grad);
    TITAN_EXPECT_EQ(value, expected);
    const Tensor dlogits = loss.backward();
    TITAN_EXPECT(dlogits.shape() == logits.shape());
    TITAN_EXPECT(dlogits.to_vector() == expected_grad.to_vector());
}

TITAN_TEST(CrossEntropyLoss, BackwardBeforeForwardThrows) {
    CrossEntropyLoss loss;
    TITAN_EXPECT_THROWS(loss.backward(), titan::Error);
}

TITAN_TEST(CrossEntropyLoss, WrongTargetCountThrows) {
    const Tensor logits = Tensor::zeros({2, 3, 5}, titan::Device::CPU);
    const std::vector<std::int32_t> targets = {0, 1, 2};
    CrossEntropyLoss loss;
    TITAN_EXPECT_THROWS(loss.forward(logits, targets), titan::Error);
}
