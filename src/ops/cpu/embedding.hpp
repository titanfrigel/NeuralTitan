#pragma once

#include <cstdint>
#include <span>

#include "core/tensor.hpp"

namespace titan::ops::cpu {

// CPU reference implementations. They do not validate; the dispatching
// functions in ops/embedding.hpp do. `shape` is the full result shape.

Tensor embedding(const Tensor& table, std::span<const std::int32_t> ids,
                 Shape shape);
void embedding_backward(Tensor& dtable_acc, std::span<const std::int32_t> ids,
                        const Tensor& dy);

}  // namespace titan::ops::cpu
