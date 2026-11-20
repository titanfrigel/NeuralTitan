#pragma once

#include "core/tensor.hpp"

namespace titan::ops::cpu {

// CPU reference implementations. They do not validate; the dispatching
// functions in ops/reduce.hpp do.

double sum(const Tensor& x);
double sum_squares(const Tensor& x);
Tensor row_max(const Tensor& x);
Tensor sum_leading(const Tensor& x);
bool all_finite(const Tensor& x);

}  // namespace titan::ops::cpu
