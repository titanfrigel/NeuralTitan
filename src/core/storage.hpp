#pragma once

#include <cstddef>

#include "core/device.hpp"

namespace titan {

// A raw, uninitialized byte buffer on one device. It is the memory a Tensor
// views; several tensors may share one Storage, so it is always held by a
// std::shared_ptr<Storage> and is neither copyable nor movable.
class Storage {
public:
    // Throws titan::Error if bytes == 0, or if device is CUDA and no CUDA
    // device is available.
    Storage(Device device, std::size_t bytes);
    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) = delete;
    Storage& operator=(Storage&&) = delete;

    void* data() { return data_; }
    const void* data() const { return data_; }
    std::size_t bytes() const { return bytes_; }
    Device device() const { return device_; }

private:
    Device device_;
    std::size_t bytes_;
    void* data_;
};

}  // namespace titan
