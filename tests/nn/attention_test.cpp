#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/attention.hpp"
#include "support/gradcheck.hpp"
#include "support/module_gradcheck.hpp"

using titan::Rng;
using titan::Tensor;
using titan::nn::Attention;
using titan::nn::ForwardContext;
using titan::nn::PositionEncoding;
using titan::test::check_module_gradients;
using titan::test::random_tensor;
using titan::test::randomize_parameters;

namespace {

constexpr float kTheta = 10000.0f;

}  // namespace

TITAN_TEST(Attention, CausalityHolds) {
    Rng rng(40);
    Attention attn("attn", 8, 4, 2, PositionEncoding::Rope, kTheta);
    randomize_parameters(attn, rng, 0.5f);
    const std::int64_t batch = 2;
    const std::int64_t t_len = 5;
    const std::int64_t dim = 8;
    Tensor x = random_tensor({batch, t_len, dim}, rng);
    ForwardContext ctx{.training = false, .pos_offset = 0};
    const std::vector<float> before = attn.forward(x, ctx).to_vector();

    // Change every feature of position t = 3 in every batch element.
    for (std::int64_t b = 0; b < batch; ++b) {
        for (std::int64_t i = 0; i < dim; ++i) {
            x.data()[(b * t_len + 3) * dim + i] += 1.0f;
        }
    }
    const std::vector<float> after = attn.forward(x, ctx).to_vector();

    for (std::int64_t b = 0; b < batch; ++b) {
        for (std::int64_t t = 0; t < t_len; ++t) {
            bool changed = false;
            for (std::int64_t i = 0; i < dim; ++i) {
                const auto k =
                    static_cast<std::size_t>((b * t_len + t) * dim + i);
                changed = changed || before[k] != after[k];
            }
            // Positions before 3 cannot see it; positions 3 and 4 do.
            TITAN_EXPECT_EQ(changed, t >= 3);
        }
    }
}

TITAN_TEST(Attention, ForwardMatchesHandComputed) {
    // One head of size 2, no RoPE, identity projections: q = k = v = x and
    // the output is softmax(causal(q·kᵀ / √2))·v.
    Attention attn("attn", 2, 1, 1, PositionEncoding::None, kTheta);
    std::vector<titan::nn::Parameter*> params;
    attn.parameters(params);
    for (titan::nn::Parameter* p : params) {
        float* w = p->value.data();
        w[0] = 1.0f;
        w[1] = 0.0f;
        w[2] = 0.0f;
        w[3] = 1.0f;
    }
    const std::vector<float> xs = {0.5f, -1.0f, 2.0f, 0.25f};
    const Tensor x = Tensor::from_vector(xs, {1, 2, 2});
    ForwardContext ctx{.training = false, .pos_offset = 0};
    const std::vector<float> y = attn.forward(x, ctx).to_vector();

    // Row 0 sees only itself, so its output is v0 = x0.
    TITAN_EXPECT_NEAR(y[0], xs[0], 1e-6f);
    TITAN_EXPECT_NEAR(y[1], xs[1], 1e-6f);
    // Row 1 sees rows 0 and 1 with weights softmax(s0, s1).
    const double s0 = (xs[2] * xs[0] + xs[3] * xs[1]) / std::sqrt(2.0);
    const double s1 = (xs[2] * xs[2] + xs[3] * xs[3]) / std::sqrt(2.0);
    const double m = std::max(s0, s1);
    const double e0 = std::exp(s0 - m);
    const double e1 = std::exp(s1 - m);
    const double p0 = e0 / (e0 + e1);
    const double p1 = e1 / (e0 + e1);
    TITAN_EXPECT_NEAR(y[2], static_cast<float>(p0 * xs[0] + p1 * xs[2]), 1e-6f);
    TITAN_EXPECT_NEAR(y[3], static_cast<float>(p0 * xs[1] + p1 * xs[3]), 1e-6f);
}

TITAN_TEST(Attention, GradCheckMHA) {
    Rng rng(41);
    Attention attn("attn", 8, 2, 2, PositionEncoding::Rope, kTheta);
    randomize_parameters(attn, rng, 0.5f);
    Tensor x = random_tensor({2, 5, 8}, rng);
    check_module_gradients(attn, x, rng);
}

TITAN_TEST(Attention, GradCheckGQA) {
    Rng rng(42);
    Attention attn("attn", 8, 4, 2, PositionEncoding::Rope, kTheta);
    randomize_parameters(attn, rng, 0.5f);
    Tensor x = random_tensor({2, 5, 8}, rng);
    check_module_gradients(attn, x, rng);
}

TITAN_TEST(Attention, GradCheckNoPE) {
    Rng rng(43);
    Attention attn("attn", 8, 4, 2, PositionEncoding::None, kTheta);
    randomize_parameters(attn, rng, 0.5f);
    Tensor x = random_tensor({2, 5, 8}, rng);
    check_module_gradients(attn, x, rng);
}

TITAN_TEST(Attention, ParameterNames) {
    Attention attn("blocks.0.attn", 8, 4, 2, PositionEncoding::Rope, kTheta);
    std::vector<titan::nn::Parameter*> params;
    attn.parameters(params);
    TITAN_EXPECT_EQ(params.size(), std::size_t{4});
    TITAN_EXPECT_EQ(params[0]->name,
                    std::string("blocks.0.attn.q_proj.weight"));
    TITAN_EXPECT_EQ(params[1]->name,
                    std::string("blocks.0.attn.k_proj.weight"));
    TITAN_EXPECT_EQ(params[2]->name,
                    std::string("blocks.0.attn.v_proj.weight"));
    TITAN_EXPECT_EQ(params[3]->name,
                    std::string("blocks.0.attn.o_proj.weight"));
    // hd = 2: q is (8 -> 8), k and v are (8 -> 2·2).
    TITAN_EXPECT(params[0]->value.shape() == titan::Shape({8, 8}));
    TITAN_EXPECT(params[1]->value.shape() == titan::Shape({4, 8}));
    TITAN_EXPECT(params[2]->value.shape() == titan::Shape({4, 8}));
    TITAN_EXPECT(params[3]->value.shape() == titan::Shape({8, 8}));
}

TITAN_TEST(Attention, InvalidHeadsThrow) {
    // dim not divisible by n_heads.
    TITAN_EXPECT_THROWS(
        Attention("a", 10, 4, 2, PositionEncoding::None, kTheta), titan::Error);
    // n_heads not divisible by n_kv_heads.
    TITAN_EXPECT_THROWS(
        Attention("a", 12, 4, 3, PositionEncoding::None, kTheta), titan::Error);
    // Odd head dimension (12 / 4 = 3) with RoPE; allowed without.
    TITAN_EXPECT_THROWS(
        Attention("a", 12, 4, 2, PositionEncoding::Rope, kTheta), titan::Error);
    Attention ok("a", 12, 4, 2, PositionEncoding::None, kTheta);
    // Non-positive head counts.
    TITAN_EXPECT_THROWS(Attention("a", 8, 0, 1, PositionEncoding::None, kTheta),
                        titan::Error);
    TITAN_EXPECT_THROWS(Attention("a", 8, 2, 0, PositionEncoding::None, kTheta),
                        titan::Error);
}

TITAN_TEST(Attention, BackwardWithoutTrainingForwardThrows) {
    Attention attn("attn", 8, 2, 2, PositionEncoding::Rope, kTheta);
    ForwardContext ctx{.training = false, .pos_offset = 0};
    const Tensor x = Tensor::zeros({1, 3, 8}, titan::Device::CPU);
    const Tensor y = attn.forward(x, ctx);
    TITAN_EXPECT_THROWS(attn.backward(y), titan::Error);
}
