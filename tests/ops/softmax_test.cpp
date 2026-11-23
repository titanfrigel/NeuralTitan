#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"
#include "ops/softmax.hpp"
#include "support/gradcheck.hpp"

using titan::Device;
using titan::Rng;
using titan::Tensor;
using titan::test::check_gradient;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

Tensor make(std::vector<float> v, titan::Shape shape) {
    return Tensor::from_vector(v, std::move(shape));
}

}  // namespace

TITAN_TEST(Softmax, RowsSumToOne) {
    Rng rng(1);
    const Tensor y = ops::softmax(random_tensor({3, 4, 7}, rng, 3.0f));
    const std::vector<float> v = y.to_vector();
    for (std::size_t r = 0; r < 12; ++r) {
        double s = 0.0;
        for (std::size_t j = 0; j < 7; ++j) {
            TITAN_EXPECT(v[r * 7 + j] > 0.0f);
            s += v[r * 7 + j];
        }
        TITAN_EXPECT_NEAR(s, 1.0, 1e-6);
    }
}

TITAN_TEST(Softmax, StableForLargeInputs) {
    const Tensor big = ops::softmax(make({1000, 1001, 1002}, {1, 3}));
    const Tensor small = ops::softmax(make({0, 1, 2}, {1, 3}));
    const std::vector<float> a = big.to_vector();
    const std::vector<float> b = small.to_vector();
    for (std::size_t j = 0; j < 3; ++j) {
        TITAN_EXPECT(std::isfinite(a[j]));
        TITAN_EXPECT_NEAR(a[j], b[j], 1e-6);
    }
}

TITAN_TEST(Softmax, CausalMaskZeros) {
    Rng rng(2);
    const Tensor y = ops::causal_softmax(random_tensor({1, 4, 4}, rng), 0);
    const std::vector<float> v = y.to_vector();
    for (std::size_t i = 0; i < 4; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < 4; ++j) {
            if (j > i) TITAN_EXPECT(v[i * 4 + j] == 0.0f);
            s += v[i * 4 + j];
        }
        TITAN_EXPECT_NEAR(s, 1.0, 1e-6);
    }
}

TITAN_TEST(Softmax, CausalWithOffset) {
    Rng rng(3);
    const Tensor y = ops::causal_softmax(random_tensor({2, 5}, rng), 3);
    const std::vector<float> v = y.to_vector();
    for (std::size_t i = 0; i < 2; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < 5; ++j) {
            if (j > i + 3) {
                TITAN_EXPECT(v[i * 5 + j] == 0.0f);
            } else {
                TITAN_EXPECT(v[i * 5 + j] > 0.0f);
            }
            s += v[i * 5 + j];
        }
        TITAN_EXPECT_NEAR(s, 1.0, 1e-6);
    }
}

TITAN_TEST(Softmax, CausalIgnoresMaskedInputs) {
    // Huge values in masked positions must not change the visible entries.
    const Tensor a = make({1, 5e30f, 2, 3}, {2, 2});
    const Tensor b = make({1, 0, 2, 3}, {2, 2});
    TITAN_EXPECT(ops::causal_softmax(a, 0).to_vector() ==
                 ops::causal_softmax(b, 0).to_vector());
}

TITAN_TEST(Softmax, GradCheck) {
    Rng rng(4);
    Tensor x = random_tensor({2, 5, 5}, rng);
    const Tensor r = random_tensor({2, 5, 5}, rng);
    const Tensor grad = ops::softmax_backward(ops::softmax(x), r);
    const auto result =
        check_gradient([&] { return ops::softmax(x); }, x, grad, r);
    TITAN_EXPECT_GRAD_OK(result);
}

TITAN_TEST(Softmax, CausalGradCheck) {
    Rng rng(5);
    Tensor x = random_tensor({2, 5, 5}, rng);
    const Tensor r = random_tensor({2, 5, 5}, rng);
    const Tensor grad = ops::softmax_backward(ops::causal_softmax(x, 0), r);
    const auto result =
        check_gradient([&] { return ops::causal_softmax(x, 0); }, x, grad, r);
    TITAN_EXPECT_GRAD_OK(result);
}

TITAN_TEST(Softmax, InvalidArgumentsThrow) {
    const Tensor v = Tensor::zeros({4}, Device::CPU);
    const Tensor m = Tensor::zeros({2, 4}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::causal_softmax(v, 0), titan::Error);
    TITAN_EXPECT_THROWS(ops::causal_softmax(m, -1), titan::Error);
    TITAN_EXPECT_THROWS(ops::causal_softmax(m, 3), titan::Error);
    TITAN_EXPECT_THROWS(ops::softmax_backward(m, v), titan::Error);
    try {
        ops::causal_softmax(m, 3);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(std::string(e.what()).find("[2, 4]") != std::string::npos);
    }
}
