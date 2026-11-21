#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops {

// Softmax. Row sums accumulate in double.

// Softmax over the last dimension, with the row maximum subtracted first.
Tensor softmax(const Tensor& x);
// Causal softmax over the last dimension of scores shaped (..., Tq, Tk).
// Entries with key index j > i + pos_offset (i is the query row) get exactly
// 0. Requires rank >= 2, pos_offset >= 0 and Tk >= Tq + pos_offset.
Tensor causal_softmax(const Tensor& scores, std::int64_t pos_offset);
// dx = y * (dy - sum_j dy_j * y_j) per row. y is the output of softmax or
// causal_softmax; masked entries of y are 0, so one function serves both.
Tensor softmax_backward(const Tensor& y, const Tensor& dy);

}  // namespace titan::ops
