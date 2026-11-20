#pragma once

#include <cstdint>
#include <span>

#include "core/tensor.hpp"

namespace titan::ops {

// Fused softmax cross-entropy over CPU tensors. A CUDA tensor throws
// "ops::cross_entropy: CUDA implementation arrives in M2".

// logits is (N, V) and targets holds N class indices, each in [0, V); the
// error names the offending target and its index. Returns the mean negative
// log-likelihood, computed in double with log-sum-exp. When dlogits_out is
// non-null it is assigned a new (N, V) tensor, the gradient of the returned
// loss: (softmax(logits) - onehot(targets)) / N.
float cross_entropy(const Tensor& logits, std::span<const std::int32_t> targets,
                    Tensor* dlogits_out);

}  // namespace titan::ops
