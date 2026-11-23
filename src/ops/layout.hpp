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

// KV-cache layout, inference only (no backward pass). A cache is a
// (B, H, Tmax, hd) tensor whose rows 0..len-1 hold the keys or values of the
// positions seen so far.

// Writes x (B, H, T, hd) into rows pos..pos+T-1 of cache (B, H, Tmax, hd):
// cache[:, :, pos:pos+T, :] = x. Mutates cache in place and leaves all other
// rows untouched. Requires matching B, H and hd, pos >= 0 and pos + T <= Tmax.
void kv_write(Tensor& cache, const Tensor& x, std::int64_t pos);
// Returns a new (B, H, len, hd) tensor holding cache[:, :, 0:len, :].
// Requires 1 <= len <= Tmax.
Tensor kv_read(const Tensor& cache, std::int64_t len);

}  // namespace titan::ops
