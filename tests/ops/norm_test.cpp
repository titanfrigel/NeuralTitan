#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/norm.hpp"
#include "support/gradcheck.hpp"

using titan::Device;
using titan::Rng;
using titan::Tensor;
using titan::test::check_gradient;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

constexpr float kEps = 1e-5f;

Tensor make(std::vector<float> v, titan::Shape shape) {
    return Tensor::from_vector(v, std::move(shape));
}

}  // namespace

TITAN_TEST(Norm, RmsNormValues) {
    const Tensor x = make({1, 2, 3, 4}, {1, 4});
    const Tensor w = make({1, 1, 1, 1}, {4});
    Tensor inv = Tensor::zeros({1}, Device::CPU);
    const std::vector<float> y = ops::rms_norm(x, w, 0.0f, &inv).to_vector();
    const double r = 1.0 / std::sqrt(7.5);
    for (std::size_t i = 0; i < 4; ++i) {
        TITAN_EXPECT_NEAR(y[i], (i + 1) * r, 1e-6);
    }
    TITAN_EXPECT((inv.shape() == titan::Shape{1}));
    TITAN_EXPECT_NEAR(inv.to_vector()[0], r, 1e-6);
}

TITAN_TEST(Norm, StatisticsShape) {
    Rng rng(10);
    const Tensor w = make(std::vector<float>(5, 1.0f), {5});
    const Tensor b = Tensor::zeros({5}, Device::CPU);
    Tensor inv = Tensor::zeros({1}, Device::CPU);
    Tensor mean = Tensor::zeros({1}, Device::CPU);
    Tensor inv_std = Tensor::zeros({1}, Device::CPU);
    ops::rms_norm(random_tensor({2, 3, 5}, rng), w, kEps, &inv);
    ops::layer_norm(random_tensor({2, 3, 5}, rng), w, b, kEps, &mean, &inv_std);
    TITAN_EXPECT((inv.shape() == titan::Shape{2, 3}));
    TITAN_EXPECT((mean.shape() == titan::Shape{2, 3}));
    TITAN_EXPECT((inv_std.shape() == titan::Shape{2, 3}));
    ops::layer_norm(random_tensor({5}, rng), w, b, kEps, &mean, nullptr);
    TITAN_EXPECT((mean.shape() == titan::Shape{1}));
}

TITAN_TEST(Norm, LayerNormZeroMeanUnitVar) {
    Rng rng(11);
    const Tensor x = random_tensor({4, 16}, rng, 3.0f);
    const Tensor w = make(std::vector<float>(16, 1.0f), {16});
    const Tensor b = Tensor::zeros({16}, Device::CPU);
    const std::vector<float> y =
        ops::layer_norm(x, w, b, 0.0f, nullptr, nullptr).to_vector();
    for (std::size_t r = 0; r < 4; ++r) {
        double mean = 0.0;
        for (std::size_t i = 0; i < 16; ++i) mean += y[r * 16 + i];
        mean /= 16.0;
        double var = 0.0;
        for (std::size_t i = 0; i < 16; ++i) {
            const double c = y[r * 16 + i] - mean;
            var += c * c;
        }
        var /= 16.0;
        TITAN_EXPECT_NEAR(mean, 0.0, 1e-5);
        TITAN_EXPECT_NEAR(var, 1.0, 1e-4);
    }
}

TITAN_TEST(Norm, RmsNormGradCheck) {
    Rng rng(12);
    Tensor x = random_tensor({3, 8}, rng);
    Tensor w = random_tensor({8}, rng);
    const Tensor r = random_tensor({3, 8}, rng);
    Tensor inv = Tensor::zeros({1}, Device::CPU);
    ops::rms_norm(x, w, kEps, &inv);
    Tensor dw = Tensor::zeros({8}, Device::CPU);
    const Tensor dx = ops::rms_norm_backward(x, w, inv, r, dw);
    const auto fwd = [&] { return ops::rms_norm(x, w, kEps, nullptr); };
    TITAN_EXPECT_GRAD_OK(check_gradient(fwd, x, dx, r));
    TITAN_EXPECT_GRAD_OK(check_gradient(fwd, w, dw, r));
}

TITAN_TEST(Norm, LayerNormGradCheck) {
    Rng rng(13);
    Tensor x = random_tensor({3, 8}, rng);
    Tensor w = random_tensor({8}, rng);
    Tensor b = random_tensor({8}, rng);
    const Tensor r = random_tensor({3, 8}, rng);
    Tensor mean = Tensor::zeros({1}, Device::CPU);
    Tensor inv_std = Tensor::zeros({1}, Device::CPU);
    ops::layer_norm(x, w, b, kEps, &mean, &inv_std);
    Tensor dw = Tensor::zeros({8}, Device::CPU);
    Tensor db = Tensor::zeros({8}, Device::CPU);
    const Tensor dx = ops::layer_norm_backward(x, w, mean, inv_std, r, dw, db);
    const auto fwd = [&] {
        return ops::layer_norm(x, w, b, kEps, nullptr, nullptr);
    };
    TITAN_EXPECT_GRAD_OK(check_gradient(fwd, x, dx, r));
    TITAN_EXPECT_GRAD_OK(check_gradient(fwd, w, dw, r));
    TITAN_EXPECT_GRAD_OK(check_gradient(fwd, b, db, r));
}

TITAN_TEST(Norm, BackwardAccumulates) {
    Rng rng(14);
    const Tensor x = random_tensor({2, 4}, rng);
    const Tensor w = random_tensor({4}, rng);
    const Tensor dy = random_tensor({2, 4}, rng);
    Tensor inv = Tensor::zeros({1}, Device::CPU);
    ops::rms_norm(x, w, kEps, &inv);
    Tensor once = Tensor::zeros({4}, Device::CPU);
    Tensor twice = Tensor::zeros({4}, Device::CPU);
    ops::rms_norm_backward(x, w, inv, dy, once);
    ops::rms_norm_backward(x, w, inv, dy, twice);
    ops::rms_norm_backward(x, w, inv, dy, twice);
    const std::vector<float> a = once.to_vector();
    const std::vector<float> c = twice.to_vector();
    for (std::size_t i = 0; i < 4; ++i) {
        TITAN_EXPECT_NEAR(c[i], 2 * a[i], 1e-6);
    }
}

TITAN_TEST(Norm, WeightShapeMismatchThrows) {
    Rng rng(15);
    const Tensor x = random_tensor({2, 4}, rng);
    const Tensor bad = random_tensor({3}, rng);
    const Tensor good = random_tensor({4}, rng);
    Tensor bad_acc = Tensor::zeros({3}, Device::CPU);
    const Tensor stat = Tensor::zeros({2}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::rms_norm(x, bad, kEps, nullptr), titan::Error);
    TITAN_EXPECT_THROWS(ops::layer_norm(x, good, bad, kEps, nullptr, nullptr),
                        titan::Error);
    TITAN_EXPECT_THROWS(ops::rms_norm_backward(x, good, stat, x, bad_acc),
                        titan::Error);
    try {
        ops::rms_norm(x, bad, kEps, nullptr);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        const std::string what = e.what();
        TITAN_EXPECT(what.find("ops::rms_norm") != std::string::npos);
        TITAN_EXPECT(what.find("[3]") != std::string::npos);
    }
}

TITAN_TEST(Norm, LayerNormBackwardAccumulates) {
    Rng rng(16);
    const Tensor x = random_tensor({2, 4}, rng);
    const Tensor w = random_tensor({4}, rng);
    const Tensor b = random_tensor({4}, rng);
    const Tensor dy = random_tensor({2, 4}, rng);
    Tensor mean = Tensor::zeros({1}, Device::CPU);
    Tensor inv_std = Tensor::zeros({1}, Device::CPU);
    ops::layer_norm(x, w, b, kEps, &mean, &inv_std);
    Tensor dw1 = Tensor::zeros({4}, Device::CPU);
    Tensor db1 = Tensor::zeros({4}, Device::CPU);
    Tensor dw2 = Tensor::zeros({4}, Device::CPU);
    Tensor db2 = Tensor::zeros({4}, Device::CPU);
    ops::layer_norm_backward(x, w, mean, inv_std, dy, dw1, db1);
    ops::layer_norm_backward(x, w, mean, inv_std, dy, dw2, db2);
    ops::layer_norm_backward(x, w, mean, inv_std, dy, dw2, db2);
    const std::vector<float> a = dw1.to_vector();
    const std::vector<float> c = dw2.to_vector();
    const std::vector<float> e = db1.to_vector();
    const std::vector<float> f = db2.to_vector();
    for (std::size_t i = 0; i < 4; ++i) {
        TITAN_EXPECT_NEAR(c[i], 2 * a[i], 1e-6);
        TITAN_EXPECT_NEAR(f[i], 2 * e[i], 1e-6);
    }
}
