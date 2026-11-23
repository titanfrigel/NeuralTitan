#pragma once

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA implementations of ops/elementwise.hpp, on CUDA tensors. They do not
// validate; the dispatching functions in ops/elementwise.cpp do. Each one
// launches one grid-stride kernel on the default stream and returns without
// waiting for it.

Tensor add(const Tensor& a, const Tensor& b);
void add_(Tensor& a, const Tensor& b);
void add_rowwise_(Tensor& y, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& x, float s);
void scale_(Tensor& x, float s);
Tensor silu(const Tensor& x);
Tensor silu_backward(const Tensor& x, const Tensor& dy);
Tensor gelu(const Tensor& x);
Tensor gelu_backward(const Tensor& x, const Tensor& dy);

}  // namespace titan::ops::cuda
