#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/rope.hpp"
#include "support/gradcheck.hpp"

using titan::Device;
using titan::Rng;
using titan::Tensor;
using titan::test::check_gradient;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

constexpr float kTheta = 10000.0f;

Tensor make(std::vector<float> v, titan::Shape shape) {
    return Tensor::from_vector(v, std::move(shape));
}

double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += double(a[i]) * b[i];
    return s;
}

// <rope(q, pq), rope(k, pk)> for single (1, 1, 1, hd) vectors.
double rotated_dot(const Tensor& q, const Tensor& k, std::int64_t pq,
                   std::int64_t pk) {
    return dot(ops::rope(q, pq, kTheta).to_vector(),
               ops::rope(k, pk, kTheta).to_vector());
}

}  // namespace

TITAN_TEST(Rope, PositionZeroIsIdentity) {
    Rng rng(1);
    const Tensor x = random_tensor({1, 2, 1, 8}, rng);
    TITAN_EXPECT(ops::rope(x, 0, kTheta).to_vector() == x.to_vector());
}

TITAN_TEST(Rope, PreservesNorm) {
    Rng rng(2);
    const Tensor x = random_tensor({2, 3, 5, 8}, rng);
    const std::vector<float> a = x.to_vector();
    const std::vector<float> b = ops::rope(x, 7, kTheta).to_vector();
    for (std::size_t v = 0; v < a.size() / 8; ++v) {
        double na = 0.0;
        double nb = 0.0;
        for (std::size_t j = 0; j < 8; ++j) {
            na += double(a[v * 8 + j]) * a[v * 8 + j];
            nb += double(b[v * 8 + j]) * b[v * 8 + j];
        }
        TITAN_EXPECT_NEAR(std::sqrt(na), std::sqrt(nb), 1e-5);
    }
}

TITAN_TEST(Rope, KnownValue) {
    const Tensor x = make({1, 0, 0, 0}, {1, 1, 1, 4});
    const std::vector<float> y = ops::rope(x, 1, kTheta).to_vector();
    TITAN_EXPECT_NEAR(y[0], std::cos(1.0), 1e-6);
    TITAN_EXPECT_NEAR(y[1], 0.0, 1e-6);
    TITAN_EXPECT_NEAR(y[2], std::sin(1.0), 1e-6);
    TITAN_EXPECT_NEAR(y[3], 0.0, 1e-6);
}

TITAN_TEST(Rope, SecondPairUsesSlowerFrequency) {
    // hd = 4, i = 1: angle = p * theta^(-1/2) = p / 100.
    const Tensor x = make({0, 1, 0, 0}, {1, 1, 1, 4});
    const std::vector<float> y = ops::rope(x, 3, kTheta).to_vector();
    TITAN_EXPECT_NEAR(y[1], std::cos(0.03), 1e-6);
    TITAN_EXPECT_NEAR(y[3], std::sin(0.03), 1e-6);
}

TITAN_TEST(Rope, PositionFollowsSequenceIndexAndOffset) {
    // Row 2 of a sequence at offset 0 rotates like that row alone at offset 2,
    // and like row 0 of the sequence at offset 2.
    Rng rng(3);
    const Tensor x = random_tensor({1, 1, 4, 6}, rng);
    const std::vector<float> all = x.to_vector();
    const std::vector<float> full = ops::rope(x, 0, kTheta).to_vector();
    const Tensor row2 = Tensor::from_vector(
        std::vector<float>(all.begin() + 12, all.begin() + 18), {1, 1, 1, 6});
    const std::vector<float> one = ops::rope(row2, 2, kTheta).to_vector();
    for (std::size_t j = 0; j < 6; ++j) {
        TITAN_EXPECT_NEAR(one[j], full[12 + j], 1e-6);
    }
}

TITAN_TEST(Rope, RelativePositionProperty) {
    Rng rng(4);
    const Tensor q = random_tensor({1, 1, 1, 8}, rng);
    const Tensor k = random_tensor({1, 1, 1, 8}, rng);
    const double base = rotated_dot(q, k, 9, 4);
    TITAN_EXPECT_NEAR(rotated_dot(q, k, 14, 9), base, 1e-4);
    TITAN_EXPECT_NEAR(rotated_dot(q, k, 5, 0), base, 1e-4);
}

TITAN_TEST(Rope, BackwardInvertsForward) {
    Rng rng(5);
    const Tensor x = random_tensor({2, 2, 3, 8}, rng);
    const std::vector<float> back =
        ops::rope_backward(ops::rope(x, 4, kTheta), 4, kTheta).to_vector();
    const std::vector<float> orig = x.to_vector();
    for (std::size_t j = 0; j < orig.size(); ++j) {
        TITAN_EXPECT_NEAR(back[j], orig[j], 1e-5);
    }
}

TITAN_TEST(Rope, GradCheck) {
    for (const std::int64_t offset : {std::int64_t{0}, std::int64_t{3}}) {
        Rng rng(6);
        Tensor x = random_tensor({2, 2, 3, 8}, rng);
        const Tensor r = random_tensor({2, 2, 3, 8}, rng);
        const Tensor grad = ops::rope_backward(r, offset, kTheta);
        const auto result = check_gradient(
            [&] { return ops::rope(x, offset, kTheta); }, x, grad, r);
        TITAN_EXPECT_GRAD_OK(result);
    }
}

TITAN_TEST(Rope, OddHeadDimThrows) {
    const Tensor x = Tensor::zeros({1, 1, 2, 3}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::rope(x, 0, kTheta), titan::Error);
    TITAN_EXPECT_THROWS(ops::rope_backward(x, 0, kTheta), titan::Error);
    try {
        ops::rope(x, 0, kTheta);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(std::string(e.what()).find("[1, 1, 2, 3]") !=
                     std::string::npos);
    }
}

TITAN_TEST(Rope, InvalidArgumentsThrow) {
    const Tensor m = Tensor::zeros({2, 4}, Device::CPU);
    const Tensor x = Tensor::zeros({1, 1, 2, 4}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::rope(m, 0, kTheta), titan::Error);
    TITAN_EXPECT_THROWS(ops::rope(x, -1, kTheta), titan::Error);
    TITAN_EXPECT_THROWS(ops::rope_backward(x, -1, kTheta), titan::Error);
}
