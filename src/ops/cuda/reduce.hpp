#pragma once

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA implementations of ops/reduce.hpp, on CUDA tensors. They do not
// validate; the dispatching functions in ops/reduce.cpp do. sum, sum_squares
// and all_finite return a host value, so they wait for the GPU (through the
// device-to-host copy of the result); row_max and sum_leading launch on the
// default stream and return the device result without waiting.

double sum(const Tensor& x);
double sum_squares(const Tensor& x);
Tensor row_max(const Tensor& x);
Tensor sum_leading(const Tensor& x);
bool all_finite(const Tensor& x);

}  // namespace titan::ops::cuda
