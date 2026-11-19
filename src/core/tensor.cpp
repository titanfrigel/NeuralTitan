#include "core/tensor.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <limits>

#include "core/error.hpp"

namespace titan {

namespace {

// Copies `bytes` between any two device/host combinations. Both sides are
// plain pointers, so only the transfer direction differs.
void copy_bytes(void* dst, Device dst_device, const void* src,
                Device src_device, std::size_t bytes) {
    if (dst_device == Device::CPU && src_device == Device::CPU) {
        std::memcpy(dst, src, bytes);
    } else if (dst_device == Device::CUDA && src_device == Device::CPU) {
        TITAN_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    } else if (dst_device == Device::CPU) {
        TITAN_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    } else {
        TITAN_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    }
}

}  // namespace

std::string to_string(const Shape& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

std::int64_t Tensor::validate_shape(const Shape& shape) {
    TITAN_CHECK(shape.size() >= 1 && shape.size() <= 4,
                "tensor rank must be 1 to 4, got shape " + to_string(shape));
    // Largest element count whose byte size still fits in int64.
    constexpr std::int64_t kMaxNumel =
        std::numeric_limits<std::int64_t>::max() /
        static_cast<std::int64_t>(sizeof(float));
    std::int64_t count = 1;
    for (std::int64_t d : shape) {
        TITAN_CHECK(d >= 1, "every dimension must be >= 1, got shape " +
                                to_string(shape));
        // count * d <= kMaxNumel, rearranged so the product never overflows.
        TITAN_CHECK(count <= kMaxNumel / d,
                    "element count overflows for shape " + to_string(shape));
        count *= d;
    }
    return count;
}

Tensor Tensor::empty(Shape shape, Device device) {
    const std::int64_t count = validate_shape(shape);
    auto storage = std::make_shared<Storage>(
        device, static_cast<std::size_t>(count) * sizeof(float));
    return Tensor(std::move(storage), 0, std::move(shape));
}

Tensor Tensor::zeros(Shape shape, Device device) {
    Tensor t = empty(std::move(shape), device);
    t.zero_();
    return t;
}

Tensor Tensor::from_vector(std::span<const float> values, Shape shape,
                           Device device) {
    const std::int64_t count = validate_shape(shape);
    TITAN_CHECK(static_cast<std::int64_t>(values.size()) == count,
                "got " + std::to_string(values.size()) + " values for shape " +
                    to_string(shape) + " (" + std::to_string(count) +
                    " elements)");
    Tensor t = empty(std::move(shape), device);
    copy_bytes(t.data(), device, values.data(), Device::CPU,
               values.size() * sizeof(float));
    return t;
}

std::int64_t Tensor::dim(int i) const {
    const int r = rank();
    TITAN_CHECK(i >= -r && i < r, "dimension " + std::to_string(i) +
                                      " out of range for shape " +
                                      to_string(shape_));
    return shape_[static_cast<std::size_t>(i < 0 ? i + r : i)];
}

std::int64_t Tensor::numel() const {
    std::int64_t count = 1;
    for (std::int64_t d : shape_) count *= d;  // validated at creation
    return count;
}

float* Tensor::data() {
    return static_cast<float*>(storage_->data()) + offset_;
}

const float* Tensor::data() const {
    return static_cast<const float*>(storage_->data()) + offset_;
}

Tensor Tensor::to(Device device) const {
    if (device == this->device()) return *this;
    Tensor out = empty(shape_, device);
    copy_bytes(out.data(), device, data(), this->device(),
               static_cast<std::size_t>(numel()) * sizeof(float));
    return out;
}

Tensor Tensor::clone() const {
    Tensor out = empty(shape_, device());
    copy_bytes(out.data(), device(), data(), device(),
               static_cast<std::size_t>(numel()) * sizeof(float));
    return out;
}

std::vector<float> Tensor::to_vector() const {
    std::vector<float> out(static_cast<std::size_t>(numel()));
    // data() already includes the view offset, so only the viewed range is
    // copied, not the whole storage.
    copy_bytes(out.data(), Device::CPU, data(), device(),
               out.size() * sizeof(float));
    return out;
}

Tensor Tensor::reshape(Shape shape) const {
    const std::int64_t count = validate_shape(shape);
    TITAN_CHECK(count == numel(), "cannot reshape " + to_string(shape_) + " (" +
                                      std::to_string(numel()) +
                                      " elements) to " + to_string(shape));
    // Same elements in the same row-major order, so the view needs no copy.
    return Tensor(storage_, offset_, std::move(shape));
}

Tensor Tensor::slice_rows(std::int64_t begin, std::int64_t end) const {
    TITAN_CHECK(begin >= 0 && begin < end && end <= shape_[0],
                "row range [" + std::to_string(begin) + ", " +
                    std::to_string(end) + ") invalid for shape " +
                    to_string(shape_));
    // One row spans numel / shape_[0] contiguous elements; skipping `begin`
    // rows is a plain offset and the selected rows stay contiguous.
    const std::int64_t row_elements = numel() / shape_[0];
    Shape sliced = shape_;
    sliced[0] = end - begin;
    return Tensor(storage_, offset_ + begin * row_elements, std::move(sliced));
}

void Tensor::zero_() {
    const std::size_t bytes = static_cast<std::size_t>(numel()) * sizeof(float);
    if (device() == Device::CUDA) {
        TITAN_CUDA_CHECK(cudaMemset(data(), 0, bytes));
    } else {
        std::memset(data(), 0, bytes);
    }
}

}  // namespace titan
