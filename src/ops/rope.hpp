#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops {

// Rotary position embedding, half-split (rotate_half) layout
// as in Hugging Face Llama checkpoints: the pairs are (i, i + hd/2).

// x is (B, H, T, hd) with hd even. Position p = pos_offset + t, and pair i
// rotates by the angle a = p * theta^(-2i/hd). Requires pos_offset >= 0.
Tensor rope(const Tensor& x, std::int64_t pos_offset, float theta);
// Gradient of rope with respect to x: the rotation by -a applied to dy.
Tensor rope_backward(const Tensor& dy, std::int64_t pos_offset, float theta);

}  // namespace titan::ops
