#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA implementations of ops/rope.hpp, on CUDA tensors. They do not
// validate; the dispatching functions in ops/rope.cpp do. Each one launches
// one grid-stride kernel on the default stream and returns without waiting
// for it.

Tensor rope(const Tensor& x, std::int64_t pos_offset, float theta);
Tensor rope_backward(const Tensor& dy, std::int64_t pos_offset, float theta);

}  // namespace titan::ops::cuda
