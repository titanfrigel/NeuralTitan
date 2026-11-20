#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops::cpu {

// CPU reference implementations. They do not validate; the dispatching
// functions in ops/softmax.hpp do.

Tensor softmax(const Tensor& x);
Tensor causal_softmax(const Tensor& scores, std::int64_t pos_offset);
Tensor softmax_backward(const Tensor& y, const Tensor& dy);

}  // namespace titan::ops::cpu
