#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA implementations of ops/layout.hpp, on CUDA tensors. They do not
// validate; the dispatching functions in ops/layout.cpp do. Each one launches
// one grid-stride kernel on the default stream and returns without waiting
// for it.

Tensor swap_dims_1_2(const Tensor& x);
Tensor repeat_kv(const Tensor& x, std::int64_t n_rep);
Tensor repeat_kv_backward(const Tensor& dy, std::int64_t n_rep);

}  // namespace titan::ops::cuda
