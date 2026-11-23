#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops {

// Head layout ops for attention. They only move or sum
// elements; see docs/math/attention.md.

// Swaps dimensions 1 and 2 of a rank-4 tensor: (A, B, C, D) -> (A, C, B, D),
// y[a, c, b, d] = x[a, b, c, d]. It is its own inverse, and therefore also its
// own backward. Attention uses it for (B, T, H, hd) <-> (B, H, T, hd).
Tensor swap_dims_1_2(const Tensor& x);

// Repeats every KV head n_rep times: x (B, Hkv, T, hd) -> y (B, Hkv·n_rep, T,
// hd) with y[b, h] = x[b, h / n_rep]. Grouped-query attention uses it to give
// each query head its KV head. Requires n_rep >= 1.
Tensor repeat_kv(const Tensor& x, std::int64_t n_rep);
// Gradient of repeat_kv: dy (B, H, T, hd) -> dx (B, H / n_rep, T, hd), where
// dx[b, g] is the sum of dy[b, g·n_rep + r] over r = 0..n_rep-1. Requires
// n_rep >= 1 dividing H. Sums accumulate in double.
Tensor repeat_kv_backward(const Tensor& dy, std::int64_t n_rep);

}  // namespace titan::ops
