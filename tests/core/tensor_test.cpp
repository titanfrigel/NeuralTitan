#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"

using titan::Device;
using titan::Shape;
using titan::Tensor;

namespace {

// Runs `fn`, expecting a titan::Error whose message contains `needle`.
template <typename Fn>
bool throws_containing(Fn&& fn, const std::string& needle) {
    try {
        fn();
    } catch (const titan::Error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

const std::vector<float> k6 = {1, 2, 3, 4, 5, 6};
const std::vector<float> k8 = {0, 1, 2, 3, 4, 5, 6, 7};

void expect_zeros(Device device) {
    const std::vector<float> values = Tensor::zeros({3, 4}, device).to_vector();
    TITAN_EXPECT_EQ(values.size(), std::size_t{12});
    for (float v : values) TITAN_EXPECT(v == 0.0f);
}

void expect_clone_is_deep(Device device) {
    Tensor original = Tensor::from_vector(k6, {2, 3}, device);
    Tensor copy = original.clone();
    TITAN_EXPECT(!copy.shares_storage_with(original));
    TITAN_EXPECT(copy.device() == device);
    copy.zero_();
    TITAN_EXPECT(original.to_vector() == k6);
    TITAN_EXPECT(copy.to_vector() == std::vector<float>(6, 0.0f));
}

}  // namespace

TITAN_TEST(Tensor, FromVectorRoundTrip) {
    Tensor t = Tensor::from_vector(k6, {2, 3});
    TITAN_EXPECT(t.to_vector() == k6);
    TITAN_EXPECT_EQ(t.numel(), std::int64_t{6});
    TITAN_EXPECT_EQ(t.dim(-1), std::int64_t{3});
    TITAN_EXPECT_EQ(t.dim(0), std::int64_t{2});
    TITAN_EXPECT_EQ(t.rank(), 2);
}

TITAN_TEST(Tensor, InvalidShapesThrow) {
    TITAN_EXPECT_THROWS(Tensor::empty({}, Device::CPU), titan::Error);
    TITAN_EXPECT_THROWS(Tensor::empty({1, 1, 1, 1, 1}, Device::CPU),
                        titan::Error);
    TITAN_EXPECT_THROWS(Tensor::empty({2, 0}, Device::CPU), titan::Error);
    TITAN_EXPECT_THROWS(Tensor::empty({-1}, Device::CPU), titan::Error);
    TITAN_EXPECT(throws_containing([] { Tensor::empty({2, 0}, Device::CPU); },
                                   "[2, 0]"));
}

TITAN_TEST(Tensor, OverflowingShapeThrows) {
    const std::int64_t big = std::int64_t{1} << 40;
    TITAN_EXPECT(throws_containing(
        [&] { Tensor::empty({big, big}, Device::CPU); }, "overflows"));
}

TITAN_TEST(Tensor, FromVectorSizeMismatchThrows) {
    const std::vector<float> five = {1, 2, 3, 4, 5};
    TITAN_EXPECT_THROWS(Tensor::from_vector(five, {2, 3}), titan::Error);
    TITAN_EXPECT(throws_containing([&] { Tensor::from_vector(five, {2, 3}); },
                                   "[2, 3]"));
    TITAN_EXPECT(
        throws_containing([&] { Tensor::from_vector(five, {2, 3}); }, "5"));
}

TITAN_TEST(Tensor, DimOutOfRangeThrows) {
    Tensor t = Tensor::zeros({2, 3}, Device::CPU);
    TITAN_EXPECT_THROWS(t.dim(2), titan::Error);
    TITAN_EXPECT_THROWS(t.dim(-3), titan::Error);
}

TITAN_TEST(Tensor, ReshapeSharesStorage) {
    Tensor t = Tensor::from_vector(k6, {2, 3});
    Tensor r = t.reshape({3, 2});
    TITAN_EXPECT(r.shares_storage_with(t));
    TITAN_EXPECT(r.shape() == (Shape{3, 2}));
    TITAN_EXPECT(r.to_vector() == k6);
    TITAN_EXPECT_THROWS(t.reshape({4, 2}), titan::Error);
}

TITAN_TEST(Tensor, SliceRowsIsAView) {
    Tensor t = Tensor::from_vector(k8, {4, 2});
    Tensor slice = t.slice_rows(1, 3);
    TITAN_EXPECT(slice.shape() == (Shape{2, 2}));
    TITAN_EXPECT(slice.shares_storage_with(t));
    TITAN_EXPECT(slice.to_vector() == (std::vector<float>{2, 3, 4, 5}));
    slice.data()[0] = 42.0f;
    TITAN_EXPECT(t.to_vector()[2] == 42.0f);
}

TITAN_TEST(Tensor, SliceOfSliceAccumulatesOffset) {
    Tensor t = Tensor::from_vector(k8, {4, 2});
    Tensor inner = t.slice_rows(1, 4).slice_rows(1, 2);
    TITAN_EXPECT(inner.to_vector() == (std::vector<float>{4, 5}));
}

TITAN_TEST(Tensor, SliceRowsOutOfRangeThrows) {
    Tensor t = Tensor::zeros({4, 2}, Device::CPU);
    TITAN_EXPECT_THROWS(t.slice_rows(2, 2), titan::Error);
    TITAN_EXPECT_THROWS(t.slice_rows(3, 1), titan::Error);
    TITAN_EXPECT_THROWS(t.slice_rows(-1, 2), titan::Error);
    TITAN_EXPECT_THROWS(t.slice_rows(0, 5), titan::Error);
}

TITAN_TEST(Tensor, ZerosAreZeroCpu) { expect_zeros(Device::CPU); }

TITAN_TEST(Tensor, ZerosAreZeroCuda) {
    TITAN_SKIP_IF_NO_CUDA();
    expect_zeros(Device::CUDA);
}

TITAN_TEST(Tensor, CpuCudaRoundTrip) {
    TITAN_SKIP_IF_NO_CUDA();
    std::vector<float> v(15);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = 0.5f * static_cast<float>(i) - 3.0f;
    }
    Tensor on_cuda = Tensor::from_vector(v, {3, 5}).to(Device::CUDA);
    TITAN_EXPECT(on_cuda.device() == Device::CUDA);
    TITAN_EXPECT(on_cuda.to(Device::CPU).to_vector() == v);
}

TITAN_TEST(Tensor, ToSameDeviceShares) {
    Tensor t = Tensor::from_vector(k6, {2, 3});
    TITAN_EXPECT(t.to(Device::CPU).shares_storage_with(t));
}

TITAN_TEST(Tensor, CloneIsDeepCpu) { expect_clone_is_deep(Device::CPU); }

TITAN_TEST(Tensor, CloneIsDeepCuda) {
    TITAN_SKIP_IF_NO_CUDA();
    expect_clone_is_deep(Device::CUDA);
}

TITAN_TEST(Tensor, CudaSliceCopiesCorrectRows) {
    TITAN_SKIP_IF_NO_CUDA();
    Tensor t = Tensor::from_vector(k8, {4, 2}, Device::CUDA);
    TITAN_EXPECT(t.slice_rows(1, 3).to_vector() ==
                 (std::vector<float>{2, 3, 4, 5}));
}

TITAN_TEST(Tensor, CudaUnavailableThrows) {
    if (titan::cuda_available()) {
        TITAN_SKIP("a CUDA device is present");
    }
    TITAN_EXPECT(throws_containing([] { Tensor::zeros({2}, Device::CUDA); },
                                   "CUDA not available"));
}
