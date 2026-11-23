#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA implementations of ops/softmax.hpp, on CUDA tensors. They do not
// validate; the dispatching functions in ops/softmax.cpp do. They launch on
// the default stream and return the device result without waiting.

Tensor softmax(const Tensor& x);
Tensor causal_softmax(const Tensor& scores, std::int64_t pos_offset);
Tensor softmax_backward(const Tensor& y, const Tensor& dy);

}  // namespace titan::ops::cuda
