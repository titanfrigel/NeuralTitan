#pragma once

#include <cstdint>
#include <span>

#include "core/tensor.hpp"

namespace titan::ops {

// Fused softmax cross-entropy. The targets live on the host; logits may be on
// the CPU or on CUDA, and the gradient is created on the logits' device.

// logits is (N, V) and targets holds N class indices, each in [0, V); the
// error names the offending target and its index. Returns the mean negative
// log-likelihood with log-sum-exp; the CPU computes it in double, CUDA sums
// each row in float and the mean over rows in double. When dlogits_out is
// non-null it is assigned a new (N, V) tensor, the gradient of the returned
// loss: (softmax(logits) - onehot(targets)) / N.
float cross_entropy(const Tensor& logits, std::span<const std::int32_t> targets,
                    Tensor* dlogits_out);

}  // namespace titan::ops
