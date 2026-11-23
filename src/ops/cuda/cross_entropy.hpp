#pragma once

#include <cstdint>
#include <span>

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA implementation of ops/cross_entropy.hpp, on a CUDA logits tensor with
// host targets. It does not validate; the dispatching function in
// ops/cross_entropy.cpp checks every target on the host before this uploads
// them. When dlogits_out is non-null it is assigned a new CUDA tensor. The
// returned loss is copied back to the host, which waits for the kernels.

float cross_entropy(const Tensor& logits, std::span<const std::int32_t> targets,
                    Tensor* dlogits_out);

}  // namespace titan::ops::cuda
