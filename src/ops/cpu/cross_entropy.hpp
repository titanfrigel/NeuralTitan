#pragma once

#include <cstdint>
#include <span>

#include "core/tensor.hpp"

namespace titan::ops::cpu {

// CPU reference implementation. It does not validate; the dispatching
// function in ops/cross_entropy.hpp does.

float cross_entropy(const Tensor& logits, std::span<const std::int32_t> targets,
                    Tensor* dlogits_out);

}  // namespace titan::ops::cpu
