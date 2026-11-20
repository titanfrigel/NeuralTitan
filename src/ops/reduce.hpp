#pragma once

#include "core/tensor.hpp"

namespace titan::ops {

// Reductions over CPU or CUDA tensors. Sums accumulate in double.
// On CUDA, sum, sum_squares and all_finite wait for the GPU.

// Sum of all elements.
double sum(const Tensor& x);
// Sum of the squares of all elements.
double sum_squares(const Tensor& x);
// Maximum over the last dimension. The result has the shape of x without its
// last dimension; a rank-1 input gives shape {1}.
Tensor row_max(const Tensor& x);
// Sum over every dimension except the last, giving shape {x.dim(-1)}.
// This is the bias gradient.
Tensor sum_leading(const Tensor& x);
// True when no element is NaN or infinite.
bool all_finite(const Tensor& x);

}  // namespace titan::ops
