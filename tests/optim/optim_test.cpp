#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/module.hpp"
#include "optim/adamw.hpp"
#include "optim/clip.hpp"
#include "optim/schedule.hpp"

using titan::Tensor;
using titan::nn::Parameter;
namespace optim = titan::optim;

namespace {

void set(Tensor& t, const std::vector<float>& v) {
    float* p = t.data();
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i];
}

}  // namespace

TITAN_TEST(AdamW, MinimizesQuadratic) {
    // L = |p|^2, so dL/dp = 2p.
    Parameter p = titan::nn::make_parameter("p", {2});
    set(p.value, {3.0f, -2.0f});
    optim::AdamWConfig cfg;
    cfg.weight_decay = 0.0f;
    optim::AdamW opt({&p}, cfg);
    for (int i = 0; i < 2000; ++i) {
        const std::vector<float> v = p.value.to_vector();
        set(p.grad, {2.0f * v[0], 2.0f * v[1]});
        opt.step(0.05f);
    }
    const std::vector<float> v = p.value.to_vector();
    TITAN_EXPECT(std::fabs(v[0]) < 1e-2f);
    TITAN_EXPECT(std::fabs(v[1]) < 1e-2f);
    TITAN_EXPECT_EQ(opt.steps(), std::int64_t{2000});
}

TITAN_TEST(AdamW, NoDecayOnRank1) {
    // With a zero gradient the moments stay zero, so the update is pure
    // decoupled decay: p -= lr * wd * p for rank >= 2 and nothing for rank 1.
    Parameter w = titan::nn::make_parameter("w", {2, 2});
    Parameter b = titan::nn::make_parameter("b", {2});
    set(w.value, {1.0f, 1.0f, 1.0f, 1.0f});
    set(b.value, {1.0f, 1.0f});
    optim::AdamWConfig cfg;
    cfg.weight_decay = 0.5f;
    optim::AdamW opt({&w, &b}, cfg);
    opt.step(0.1f);
    for (float x : w.value.to_vector()) TITAN_EXPECT_NEAR(x, 0.95, 1e-6);
    for (float x : b.value.to_vector()) TITAN_EXPECT_EQ(x, 1.0f);
}

TITAN_TEST(Clip, ScalesWhenAboveMax) {
    Parameter a = titan::nn::make_parameter("a", {2});
    Parameter b = titan::nn::make_parameter("b", {1});
    set(a.grad, {3.0f, 0.0f});
    set(b.grad, {4.0f});
    const double norm = optim::clip_grad_norm({&a, &b}, 2.5);
    TITAN_EXPECT_EQ(norm, 5.0);
    const std::vector<float> ga = a.grad.to_vector();
    TITAN_EXPECT_NEAR(ga[0], 1.5, 1e-6);
    TITAN_EXPECT_NEAR(ga[1], 0.0, 1e-6);
    TITAN_EXPECT_NEAR(b.grad.to_vector()[0], 2.0, 1e-6);
}

TITAN_TEST(Clip, NoopWhenBelow) {
    Parameter a = titan::nn::make_parameter("a", {2});
    set(a.grad, {3.0f, 4.0f});
    const double norm = optim::clip_grad_norm({&a}, 10.0);
    TITAN_EXPECT_EQ(norm, 5.0);
    TITAN_EXPECT(a.grad.to_vector() == (std::vector<float>{3.0f, 4.0f}));
}

TITAN_TEST(Clip, NonFiniteNormIsReturnedUnscaled) {
    Parameter a = titan::nn::make_parameter("a", {2});
    set(a.grad, {1.0f, std::numeric_limits<float>::infinity()});
    const double norm = optim::clip_grad_norm({&a}, 1.0);
    TITAN_EXPECT(!std::isfinite(norm));
    TITAN_EXPECT(a.grad.to_vector()[0] == 1.0f);
}

TITAN_TEST(Schedule, WarmupAndCosine) {
    const float hi = 1e-3f;
    const float lo = 1e-4f;
    TITAN_EXPECT_NEAR(optim::lr_at(0, 10, 100, hi, lo), 1e-4, 1e-9);
    TITAN_EXPECT_NEAR(optim::lr_at(9, 10, 100, hi, lo), 1e-3, 1e-9);
    TITAN_EXPECT_NEAR(optim::lr_at(99, 10, 100, hi, lo), 1e-4, 1e-9);
    TITAN_EXPECT_NEAR(optim::lr_at(500, 10, 100, hi, lo), 1e-4, 1e-9);
    // Warmup is increasing, decay never increases.
    for (std::int64_t s = 1; s < 10; ++s) {
        TITAN_EXPECT(optim::lr_at(s, 10, 100, hi, lo) >
                     optim::lr_at(s - 1, 10, 100, hi, lo));
    }
    for (std::int64_t s = 10; s < 100; ++s) {
        TITAN_EXPECT(optim::lr_at(s, 10, 100, hi, lo) <=
                     optim::lr_at(s - 1, 10, 100, hi, lo));
    }
}

TITAN_TEST(Schedule, RejectsBadArguments) {
    TITAN_EXPECT_THROWS(optim::lr_at(0, 0, 10, 1.0f, 0.1f), titan::Error);
    TITAN_EXPECT_THROWS(optim::lr_at(0, 10, 10, 1.0f, 0.1f), titan::Error);
}
