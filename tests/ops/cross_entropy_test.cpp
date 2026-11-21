#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/cross_entropy.hpp"
#include "support/gradcheck.hpp"

using titan::Device;
using titan::Rng;
using titan::Tensor;
using titan::test::check_gradient;
using titan::test::random_tensor;
namespace ops = titan::ops;

TITAN_TEST(CrossEntropy, UniformLogitsGiveLogV) {
    const Tensor logits = Tensor::zeros({3, 65}, Device::CPU);
    const std::vector<std::int32_t> targets = {0, 17, 64};
    TITAN_EXPECT_NEAR(ops::cross_entropy(logits, targets, nullptr),
                      std::log(65.0), 1e-5);
}

TITAN_TEST(CrossEntropy, StableForLargeLogits) {
    const std::vector<float> big = {1000, 1001, 1002, -1000, 0, 1000};
    const Tensor logits = Tensor::from_vector(big, {2, 3});
    const std::vector<std::int32_t> targets = {0, 1};
    Tensor d = Tensor::zeros({1, 1}, Device::CPU);
    const float loss = ops::cross_entropy(logits, targets, &d);
    TITAN_EXPECT(std::isfinite(loss));
    // Row 0 is the row (0, 1, 2) shifted by 1000, target 0: loss
    // lse - z_target = 2 + log(e^-2 + e^-1 + 1) (relative to the target).
    const double row0 = 2.0 + std::log(std::exp(-2.0) + std::exp(-1.0) + 1.0);
    // Row 1: z = (-1000, 0, 1000), target 1, loss = 1000.
    TITAN_EXPECT_NEAR(loss, (row0 + 1000.0) / 2.0, 1e-3);
    for (float g : d.to_vector()) TITAN_EXPECT(std::isfinite(g));
}

TITAN_TEST(CrossEntropy, GradientRowsSumToZero) {
    Rng rng(6);
    const Tensor logits = random_tensor({4, 7}, rng, 2.0f);
    const std::vector<std::int32_t> targets = {0, 6, 3, 3};
    Tensor d = Tensor::zeros({1, 1}, Device::CPU);
    ops::cross_entropy(logits, targets, &d);
    TITAN_EXPECT(d.shape() == titan::Shape({4, 7}));
    const std::vector<float> g = d.to_vector();
    for (std::size_t i = 0; i < 4; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < 7; ++j) s += g[i * 7 + j];
        TITAN_EXPECT_NEAR(s, 0.0, 1e-6);
    }
}

TITAN_TEST(CrossEntropy, GradCheck) {
    Rng rng(7);
    Tensor logits = random_tensor({4, 7}, rng, 2.0f);
    const std::vector<std::int32_t> targets = {1, 0, 6, 3};
    Tensor d = Tensor::zeros({1, 1}, Device::CPU);
    ops::cross_entropy(logits, targets, &d);
    // The loss itself is the scalar: a 1-element output with r = 1.
    const Tensor r = Tensor::from_vector(std::vector<float>{1.0f}, {1});
    const auto result = check_gradient(
        [&] {
            const float loss = ops::cross_entropy(logits, targets, nullptr);
            return Tensor::from_vector(std::vector<float>{loss}, {1});
        },
        logits, d, r);
    TITAN_EXPECT_GRAD_OK(result);
}

TITAN_TEST(CrossEntropy, TargetCountMismatchThrows) {
    const Tensor logits = Tensor::zeros({3, 5}, Device::CPU);
    const std::vector<std::int32_t> two = {0, 1};
    TITAN_EXPECT_THROWS(ops::cross_entropy(logits, two, nullptr), titan::Error);
}

TITAN_TEST(CrossEntropy, InvalidArgumentsThrow) {
    const Tensor logits = Tensor::zeros({2, 5}, Device::CPU);
    for (std::int32_t bad : {-1, 5}) {
        const std::vector<std::int32_t> targets = {0, bad};
        try {
            ops::cross_entropy(logits, targets, nullptr);
            TITAN_EXPECT(false);
        } catch (const titan::Error& e) {
            TITAN_EXPECT(std::string(e.what()).find(
                             "target " + std::to_string(bad) + " at index 1") !=
                         std::string::npos);
        }
    }
    const std::vector<std::int32_t> one = {0};
    TITAN_EXPECT_THROWS(
        ops::cross_entropy(Tensor::zeros({5}, Device::CPU), one, nullptr),
        titan::Error);
}
