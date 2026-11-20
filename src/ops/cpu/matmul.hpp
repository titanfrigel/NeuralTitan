#pragma once

#include "core/tensor.hpp"

namespace titan::ops::cpu {

// CPU matmul kernels. They do not validate; ops/matmul.hpp does.
//
// Both take a of shape (..., A0, A1) and b of shape (..., B0, B1) with equal
// leading (batch) dimensions, where op(a) is (M, K) and op(b) is (K, N) after
// the optional transposes, and out is (..., M, N).

// out += op(a)·op(b). Parallel over batch × M rows; see docs/math/matmul.md
// for the loop order and the fixed accumulation order.
void matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a,
                bool transpose_b);

// out = op(a)·op(b) by a plain triple loop with a double accumulator per
// element. The reference the fast kernel is tested against.
void matmul_naive(Tensor& out, const Tensor& a, const Tensor& b,
                  bool transpose_a, bool transpose_b);

}  // namespace titan::ops::cpu
