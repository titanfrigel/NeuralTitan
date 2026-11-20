#pragma once

#include <initializer_list>
#include <string>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/tensor.hpp"

namespace titan::ops::detail {

// Throws exactly "ops::<op>: CUDA implementation arrives in M2" when any
// tensor is not on the CPU. Until M2 the CUDA backend does not exist.
inline void require_cpu(const char* op,
                        std::initializer_list<const Tensor*> tensors) {
    for (const Tensor* t : tensors) {
        if (t->device() != Device::CPU) {
            throw_error(__FILE__, __LINE__,
                        std::string("ops::") + op +
                            ": CUDA implementation arrives in M2");
        }
    }
}

// Throws naming the op and both shapes unless the shapes are equal.
inline void require_same_shape(const char* op, const Tensor& a,
                               const Tensor& b) {
    TITAN_CHECK(a.shape() == b.shape(),
                std::string("ops::") + op + ": shape mismatch, " +
                    to_string(a.shape()) + " vs " + to_string(b.shape()));
}

}  // namespace titan::ops::detail
