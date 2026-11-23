#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops::cpu {

// CPU reference implementation. It does not validate; the dispatching
// function in ops/adamw.hpp does.

void adamw_update(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, std::int64_t step);

}  // namespace titan::ops::cpu
