#pragma once

#include "core/tensor.hpp"

namespace titan::ops {

// Matrix product of CPU tensors of rank 2 to 4. The leading dimensions are
// batch dimensions and must be equal in a and b. With the optional transposes
// applied to the last two dimensions, op(a) is (..., M, K) and op(b) is
// (..., K, N); the result is (..., M, N).
// A CUDA tensor throws "ops::matmul: CUDA implementation arrives in M2".
Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a = false,
              bool transpose_b = false);

// out += op(a)·op(b). out must already have the result shape. Used to
// accumulate weight gradients.
void matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a,
                bool transpose_b);

}  // namespace titan::ops
