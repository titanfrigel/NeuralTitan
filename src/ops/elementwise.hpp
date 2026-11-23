#pragma once

#include "core/tensor.hpp"

namespace titan::ops {

// Elementwise operations on CPU tensors. Two-input ops need equal shapes.
// A CUDA tensor throws "ops::<name>: CUDA implementation arrives in M2".

// a + b.
Tensor add(const Tensor& a, const Tensor& b);
// a += b, in place.
void add_(Tensor& a, const Tensor& b);
// y[r, :] += b for every row r of y, in place. y has last dimension D and b
// is {D}; this is the broadcast add of a bias.
void add_rowwise_(Tensor& y, const Tensor& b);
// a * b, elementwise.
Tensor mul(const Tensor& a, const Tensor& b);
// s * x.
Tensor scale(const Tensor& x, float s);
// x *= s, in place.
void scale_(Tensor& x, float s);

// SiLU: x * sigmoid(x). The backward returns dL/dx given x and dL/dy.
Tensor silu(const Tensor& x);
Tensor silu_backward(const Tensor& x, const Tensor& dy);

// GELU, tanh approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
Tensor gelu(const Tensor& x);
Tensor gelu_backward(const Tensor& x, const Tensor& dy);

}  // namespace titan::ops
