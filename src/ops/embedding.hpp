#pragma once

#include <cstdint>
#include <span>

#include "core/tensor.hpp"

namespace titan::ops {

// Embedding lookup over CPU tensors. A CUDA tensor throws
// "ops::<name>: CUDA implementation arrives in M2". Every id must be in
// [0, V); the error names the offending id and its index.

// table is (V, D). out_shape is the layout of the ids (for example {B, T}) and
// its element count must equal ids.size(). The result has shape
// out_shape + {D}, whose rank must be at most 4: row k of the flattened result
// is table row ids[k].
Tensor embedding(const Tensor& table, std::span<const std::int32_t> ids,
                 Shape out_shape);
// Scatter-add of the rows of dy into dtable_acc (shaped like the table):
// dtable_acc[ids[k]] += dy[k]. Repeated ids accumulate. dy has ids.size() * D
// elements with last dimension D. Per-element sums over repeated ids are
// accumulated in double.
void embedding_backward(Tensor& dtable_acc, std::span<const std::int32_t> ids,
                        const Tensor& dy);

}  // namespace titan::ops
