#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"
#include "ops/adamw.hpp"

using titan::Device;
using titan::Tensor;
namespace ops = titan::ops;

namespace {

Tensor make(std::vector<float> v, titan::Shape shape) {
    return Tensor::from_vector(v, std::move(shape));
}

ops::AdamWHyper hyper(float wd, std::int64_t step) {
    return {0.1f, 0.9f, 0.999f, 1e-8f, wd, step};
}

}  // namespace

TITAN_TEST(AdamW, FirstStepMatchesFormula) {
    // At step 1 the bias-corrected moments are m_hat = g and v_hat = g^2, so
    // the update is lr * (g / (|g| + eps) + wd * p) = lr * (sign(g) + wd * p)
    // for g != 0, and lr * wd * p for g = 0.
    Tensor p = make({1.0f, -2.0f, 0.5f}, {3});
    const Tensor g = make({0.1f, -0.2f, 0.0f}, {3});
    Tensor m = Tensor::zeros({3}, Device::CPU);
    Tensor v = Tensor::zeros({3}, Device::CPU);
    ops::adamw_update(p, g, m, v, hyper(0.01f, 1));

    const std::vector<float> pv = p.to_vector();
    TITAN_EXPECT_NEAR(pv[0], 1.0 - 0.1 * (1.0 + 0.01 * 1.0), 1e-6);
    TITAN_EXPECT_NEAR(pv[1], -2.0 - 0.1 * (-1.0 + 0.01 * -2.0), 1e-6);
    TITAN_EXPECT_NEAR(pv[2], 0.5 - 0.1 * (0.01 * 0.5), 1e-6);

    const std::vector<float> mv = m.to_vector();
    const std::vector<float> vv = v.to_vector();
    TITAN_EXPECT_NEAR(mv[0], 0.1 * 0.1, 1e-7);
    TITAN_EXPECT_NEAR(mv[1], 0.1 * -0.2, 1e-7);
    TITAN_EXPECT_NEAR(vv[0], 0.001 * 0.01, 1e-9);
    TITAN_EXPECT_NEAR(vv[1], 0.001 * 0.04, 1e-9);
}

TITAN_TEST(AdamW, SecondStepUsesBiasCorrection) {
    Tensor p = make({1.0f}, {1});
    const Tensor g = make({0.5f}, {1});
    Tensor m = Tensor::zeros({1}, Device::CPU);
    Tensor v = Tensor::zeros({1}, Device::CPU);
    ops::adamw_update(p, g, m, v, hyper(0.0f, 1));
    ops::adamw_update(p, g, m, v, hyper(0.0f, 2));
    // A constant gradient keeps m_hat = g and v_hat = g^2 at every step, so
    // each step moves the parameter by lr * g / (|g| + eps) = lr.
    TITAN_EXPECT_NEAR(p.to_vector()[0], 1.0 - 0.2, 1e-6);
}

TITAN_TEST(AdamW, WeightDecayShrinksWithZeroGrad) {
    Tensor p = make({2.0f, -4.0f}, {2});
    const Tensor g = Tensor::zeros({2}, Device::CPU);
    Tensor m = Tensor::zeros({2}, Device::CPU);
    Tensor v = Tensor::zeros({2}, Device::CPU);
    ops::adamw_update(p, g, m, v, hyper(0.5f, 1));
    // p -= lr * wd * p  =>  p *= 1 - 0.1 * 0.5.
    const std::vector<float> pv = p.to_vector();
    TITAN_EXPECT_NEAR(pv[0], 2.0 * 0.95, 1e-6);
    TITAN_EXPECT_NEAR(pv[1], -4.0 * 0.95, 1e-6);
}

TITAN_TEST(AdamW, InvalidArgumentsThrow) {
    Tensor p = Tensor::zeros({3}, Device::CPU);
    const Tensor g = Tensor::zeros({3}, Device::CPU);
    Tensor m = Tensor::zeros({3}, Device::CPU);
    Tensor v = Tensor::zeros({3}, Device::CPU);
    Tensor wrong = Tensor::zeros({4}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::adamw_update(p, g, m, v, hyper(0.0f, 0)),
                        titan::Error);
    TITAN_EXPECT_THROWS(ops::adamw_update(p, g, wrong, v, hyper(0.0f, 1)),
                        titan::Error);
    TITAN_EXPECT_THROWS(ops::adamw_update(p, wrong, m, v, hyper(0.0f, 1)),
                        titan::Error);
    TITAN_EXPECT_THROWS(ops::adamw_update(p, g, m, wrong, hyper(0.0f, 1)),
                        titan::Error);
}

TITAN_TEST(AdamW, CudaTensorThrowsM2Message) {
    TITAN_SKIP_IF_NO_CUDA();
    Tensor p = Tensor::zeros({2}, Device::CUDA);
    const Tensor g = Tensor::zeros({2}, Device::CUDA);
    Tensor m = Tensor::zeros({2}, Device::CUDA);
    Tensor v = Tensor::zeros({2}, Device::CUDA);
    try {
        ops::adamw_update(p, g, m, v, hyper(0.0f, 1));
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(std::string(e.what()).find(
                         "ops::adamw_update: CUDA implementation arrives in "
                         "M2") != std::string::npos);
    }
}
