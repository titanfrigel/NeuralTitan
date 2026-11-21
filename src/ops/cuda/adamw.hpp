#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA implementation of ops/adamw.hpp, on CUDA tensors. It does not
// validate; the dispatching function in ops/adamw.cpp does. It launches one
// grid-stride kernel on the default stream and returns without waiting for it.

void adamw_update(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, std::int64_t step);

}  // namespace titan::ops::cuda
