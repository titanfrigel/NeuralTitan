#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/elementwise.hpp"
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

TITAN_TEST(Elementwise, AddMulScale) {
    const Tensor a = make({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor b = make({6, 5, 4, 3, 2, 1}, {2, 3});
    TITAN_EXPECT(ops::add(a, b).to_vector() ==
                 (std::vector<float>{7, 7, 7, 7, 7, 7}));
    TITAN_EXPECT(ops::mul(a, b).to_vector() ==
                 (std::vector<float>{6, 10, 12, 12, 10, 6}));
    TITAN_EXPECT(ops::scale(a, 0.5f).to_vector() ==
                 (std::vector<float>{0.5f, 1, 1.5f, 2, 2.5f, 3}));
    TITAN_EXPECT(ops::add(a, b).shape() == (titan::Shape{2, 3}));
}

TITAN_TEST(Elementwise, ScaleInPlace) {
    Tensor a = make({1, 2, 3, 4}, {2, 2});
    ops::scale_(a, -0.5f);
    TITAN_EXPECT(a.to_vector() == (std::vector<float>{-0.5f, -1, -1.5f, -2}));
}

TITAN_TEST(Elementwise, AddInPlaceAccumulates) {
    Tensor a = make({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor b = make({1, 1, 1, 1, 1, 1}, {2, 3});
    ops::add_(a, b);
    ops::add_(a, b);
    TITAN_EXPECT(a.to_vector() == (std::vector<float>{3, 4, 5, 6, 7, 8}));
}

TITAN_TEST(Elementwise, ShapeMismatchThrows) {
    const Tensor a = Tensor::zeros({2, 3}, Device::CPU);
    const Tensor b = Tensor::zeros({3, 2}, Device::CPU);
    try {
        ops::add(a, b);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        const std::string msg = e.what();
        TITAN_EXPECT(msg.find("[2, 3]") != std::string::npos);
        TITAN_EXPECT(msg.find("[3, 2]") != std::string::npos);
    }
    TITAN_EXPECT_THROWS(ops::mul(a, b), titan::Error);
    TITAN_EXPECT_THROWS(ops::silu_backward(a, b), titan::Error);
}

TITAN_TEST(Elementwise, SiluValues) {
    const Tensor y = ops::silu(make({0.0f, 1.0f}, {2}));
    const auto v = y.to_vector();
    TITAN_EXPECT_NEAR(v[0], 0.0f, 1e-6f);
    TITAN_EXPECT_NEAR(v[1], 0.7310586f, 1e-6f);
}

TITAN_TEST(Elementwise, GeluValues) {
    const Tensor y = ops::gelu(make({0.0f, 1.0f}, {2}));
    const auto v = y.to_vector();
    TITAN_EXPECT_NEAR(v[0], 0.0f, 1e-6f);
    TITAN_EXPECT_NEAR(v[1], 0.8411920f, 1e-6f);
}

TITAN_TEST(Elementwise, SiluGradCheck) {
    Rng rng(7);
    Tensor x = random_tensor({3, 5}, rng);
    const Tensor r = random_tensor({3, 5}, rng);
    const Tensor grad = ops::silu_backward(x, r);
    const auto result =
        check_gradient([&] { return ops::silu(x); }, x, grad, r);
    TITAN_EXPECT_GRAD_OK(result);
}

TITAN_TEST(Elementwise, GeluGradCheck) {
    Rng rng(8);
    Tensor x = random_tensor({3, 5}, rng);
    const Tensor r = random_tensor({3, 5}, rng);
    const Tensor grad = ops::gelu_backward(x, r);
    const auto result =
        check_gradient([&] { return ops::gelu(x); }, x, grad, r);
    TITAN_EXPECT_GRAD_OK(result);
}

TITAN_TEST(Elementwise, AddRowwiseValues) {
    Tensor y = make({1, 2, 3, 4, 5, 6}, {2, 3});
    ops::add_rowwise_(y, make({10, 20, 30}, {3}));
    TITAN_EXPECT(y.to_vector() == std::vector<float>({11, 22, 33, 14, 25, 36}));
    Tensor y3 = make({1, 2, 3, 4}, {2, 1, 2});
    ops::add_rowwise_(y3, make({1, -1}, {2}));
    TITAN_EXPECT(y3.to_vector() == std::vector<float>({2, 1, 4, 3}));
}

TITAN_TEST(Elementwise, AddRowwiseShapeMismatchThrows) {
    Tensor y = Tensor::zeros({2, 3}, Device::CPU);
    const Tensor b = Tensor::zeros({2}, Device::CPU);
    try {
        ops::add_rowwise_(y, b);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        const std::string msg = e.what();
        TITAN_EXPECT(msg.find("ops::add_rowwise_") != std::string::npos);
        TITAN_EXPECT(msg.find("[2, 3]") != std::string::npos);
        TITAN_EXPECT(msg.find("[2]") != std::string::npos);
    }
    const Tensor b2 = Tensor::zeros({1, 3}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::add_rowwise_(y, b2), titan::Error);
}
