#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/device.hpp"
#include "core/storage.hpp"

namespace titan {

// Tensor dimensions, outermost first. Rank 1 to 4, every dimension >= 1.
using Shape = std::vector<std::int64_t>;

// "[2, 3]".
std::string to_string(const Shape& shape);

// A dense FP32 tensor: a shape plus a window into a shared Storage.
//
// Every tensor is row-major and contiguous: element (i, j) of a {R, C} tensor
// lives at offset_ + i * C + j. Because views (reshape, slice_rows) only ever
// cut whole rows off the outermost dimension, they stay contiguous and need no
// strides; the window is fully described by offset_ and shape_.
//
// Copying a Tensor is cheap: the copy shares the Storage. Use clone() for a
// deep copy.
class Tensor {
public:
    // Uninitialized contents. Throws titan::Error for an invalid shape, or for
    // Device::CUDA without a GPU.
    static Tensor empty(Shape shape, Device device);
    static Tensor zeros(Shape shape, Device device);
    // values.size() must equal the number of elements of `shape`.
    static Tensor from_vector(std::span<const float> values, Shape shape,
                              Device device = Device::CPU);

    const Shape& shape() const { return shape_; }
    int rank() const { return static_cast<int>(shape_.size()); }
    // Negative `i` counts from the end: dim(-1) is the innermost dimension.
    std::int64_t dim(int i) const;
    std::int64_t numel() const;

    Device device() const { return storage_->device(); }
    // Pointer to the first element of this view; a device pointer for CUDA.
    float* data();
    const float* data() const;

    // Returns *this (shared storage) when the device already matches,
    // otherwise a copy on `device`.
    Tensor to(Device device) const;
    // Always a deep copy on the same device.
    Tensor clone() const;
    // Copies the viewed elements to the host, from either device.
    std::vector<float> to_vector() const;

    // Views sharing storage with this tensor.
    Tensor reshape(Shape shape) const;
    // Rows [begin, end) along dimension 0.
    Tensor slice_rows(std::int64_t begin, std::int64_t end) const;

    // Sets every element to zero in place.
    void zero_();

    bool shares_storage_with(const Tensor& other) const {
        return storage_ == other.storage_;
    }

private:
    Tensor(std::shared_ptr<Storage> storage, std::int64_t offset, Shape shape)
        : storage_(std::move(storage)),
          offset_(offset),
          shape_(std::move(shape)) {}

    // Throws if the shape is not rank 1-4 with positive dimensions, or if its
    // element count overflows. Returns the element count.
    static std::int64_t validate_shape(const Shape& shape);

    std::shared_ptr<Storage> storage_;
    std::int64_t offset_;  // in elements, not bytes
    Shape shape_;
};

}  // namespace titan
