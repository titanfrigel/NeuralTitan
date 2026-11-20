#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Tensor;
using titan::test::check_parity;

namespace {

Tensor on_cuda(std::vector<float> v) {
    const auto n = static_cast<std::int64_t>(v.size());
    return Tensor::from_vector(v, {n}, Device::CUDA);
}

Tensor on_cpu(std::vector<float> v) {
    const auto n = static_cast<std::int64_t>(v.size());
    return Tensor::from_vector(v, {n});
}

}  // namespace

TITAN_TEST(Parity, RandomTensorIsSeeded) {
    const Tensor a = titan::test::random_tensor({4, 3}, 42);
    const Tensor b = titan::test::random_tensor({4, 3}, 42);
    const Tensor c = titan::test::random_tensor({4, 3}, 43);
    TITAN_EXPECT(a.device() == Device::CPU);
    TITAN_EXPECT(a.to_vector() == b.to_vector());
    TITAN_EXPECT(a.to_vector() != c.to_vector());
}

TITAN_TEST(Parity, AcceptsValuesWithinTolerance) {
    TITAN_REQUIRE_CUDA();
    // |c - g| <= atol + rtol * |c|: 1e-5 + 1e-4 * 100 = 0.01001 at index 1.
    const auto r = check_parity(on_cpu({0.0f, 100.0f}),
                                on_cuda({5e-6f, 100.01f}), 1e-5, 1e-4);
    TITAN_EXPECT(r.ok);
    TITAN_EXPECT(r.detail.empty());
}

TITAN_TEST(Parity, ReportsWorstElement) {
    TITAN_REQUIRE_CUDA();
    const auto r = check_parity(on_cpu({1.0f, 2.0f, 3.0f}),
                                on_cuda({1.1f, 2.0f, 3.5f}), 1e-5, 1e-4);
    TITAN_EXPECT(!r.ok);
    TITAN_EXPECT(r.detail.find("2 of 3 elements") != std::string::npos);
    TITAN_EXPECT(r.detail.find("worst at index 2") != std::string::npos);
    TITAN_EXPECT(r.detail.find("cpu 3 vs cuda 3.5") != std::string::npos);
}

TITAN_TEST(Parity, NanFails) {
    TITAN_REQUIRE_CUDA();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto r =
        check_parity(on_cpu({1.0f, 2.0f}), on_cuda({1.0f, nan}), 1e-5, 1e-4);
    TITAN_EXPECT(!r.ok);
    TITAN_EXPECT(r.detail.find("worst at index 1") != std::string::npos);
}

TITAN_TEST(Parity, RejectsShapeAndDeviceMismatch) {
    TITAN_REQUIRE_CUDA();
    const Tensor c = Tensor::zeros({2, 3}, Device::CPU);
    const auto shape =
        check_parity(c, Tensor::zeros({3, 2}, Device::CUDA), 1e-5, 1e-4);
    TITAN_EXPECT(!shape.ok);
    TITAN_EXPECT(shape.detail.find("[2, 3] vs [3, 2]") != std::string::npos);
    const auto device = check_parity(c, c, 1e-5, 1e-4);
    TITAN_EXPECT(!device.ok);
    TITAN_EXPECT(device.detail.find("got cpu and cpu") != std::string::npos);
}
