#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops::cpu {

// CPU reference implementations. They do not validate; the dispatching
// functions in ops/layout.hpp do.

Tensor swap_dims_1_2(const Tensor& x);
Tensor repeat_kv(const Tensor& x, std::int64_t n_rep);
Tensor repeat_kv_backward(const Tensor& dy, std::int64_t n_rep);

}  // namespace titan::ops::cpu
