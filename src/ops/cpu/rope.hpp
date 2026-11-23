#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops::cpu {

// CPU reference implementations. They do not validate; the dispatching
// functions in ops/rope.hpp do.

Tensor rope(const Tensor& x, std::int64_t pos_offset, float theta);
Tensor rope_backward(const Tensor& dy, std::int64_t pos_offset, float theta);

}  // namespace titan::ops::cpu
