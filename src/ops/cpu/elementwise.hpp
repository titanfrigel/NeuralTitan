#pragma once

#include "core/tensor.hpp"

namespace titan::ops::cpu {

// CPU reference implementations. They do not validate; the dispatching
// functions in ops/elementwise.hpp do.

Tensor add(const Tensor& a, const Tensor& b);
void add_(Tensor& a, const Tensor& b);
void add_rowwise_(Tensor& y, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& x, float s);
Tensor silu(const Tensor& x);
Tensor silu_backward(const Tensor& x, const Tensor& dy);
Tensor gelu(const Tensor& x);
Tensor gelu_backward(const Tensor& x, const Tensor& dy);

}  // namespace titan::ops::cpu
