#pragma once

#include <cstdint>
#include <span>

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA implementations of ops/embedding.hpp, on CUDA tensors with host ids.
// They do not validate; the dispatching functions in ops/embedding.cpp check
// every id on the host before these upload them. `shape` is the full result
// shape. They launch on the default stream; the id upload is a synchronous
// host-to-device copy.

Tensor embedding(const Tensor& table, std::span<const std::int32_t> ids,
                 Shape shape);
void embedding_backward(Tensor& dtable_acc, std::span<const std::int32_t> ids,
                        const Tensor& dy);

}  // namespace titan::ops::cuda
