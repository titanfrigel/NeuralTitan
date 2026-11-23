#pragma once

#include <initializer_list>
#include <string>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/tensor.hpp"

namespace titan::ops::detail {

// The device backend an op runs on.
enum class Backend { cpu, cuda };

// Returns the backend of the device every tensor is on. Throws
// "ops::<op>: tensors are on different devices (<a> vs <b>)" otherwise, where
// <a> is the first tensor's device and <b> the device of the first tensor in
// argument order that differs from it. Called before any other check, so a
// mixed call never reaches a kernel or a host read of a device pointer.
inline Backend route(const char* op,
                     std::initializer_list<const Tensor*> tensors) {
    TITAN_CHECK(tensors.size() > 0,
                std::string("ops::") + op + ": route needs a tensor");
    const Device first = (*tensors.begin())->device();
    for (const Tensor* t : tensors) {
        if (t->device() != first) {
            throw_error(__FILE__, __LINE__,
                        std::string("ops::") + op +
                            ": tensors are on different devices (" +
                            std::string(to_string(first)) + " vs " +
                            std::string(to_string(t->device())) + ")");
        }
    }
    return first == Device::CPU ? Backend::cpu : Backend::cuda;
}

// Throws naming the op and both shapes unless the shapes are equal.
inline void require_same_shape(const char* op, const Tensor& a,
                               const Tensor& b) {
    TITAN_CHECK(a.shape() == b.shape(),
                std::string("ops::") + op + ": shape mismatch, " +
                    to_string(a.shape()) + " vs " + to_string(b.shape()));
}

}  // namespace titan::ops::detail
