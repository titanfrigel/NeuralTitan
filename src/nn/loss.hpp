#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "core/tensor.hpp"

namespace titan::nn {

// Mean cross-entropy of next-token logits against target ids. Not a Module:
// its second input is integer ids and its output is a scalar. forward()
// flattens logits (..., V), typically (B, T, V), to (N, V) with N the number
// of targets, and caches the gradient of the mean loss; backward() returns it
// in the logits' shape. See docs/math/cross-entropy.md.
class CrossEntropyLoss {
public:
    float forward(const Tensor& logits, std::span<const std::int32_t> targets);
    // dL/dlogits of the last forward. Throws before any forward.
    Tensor backward() const;

private:
    std::optional<Tensor> dlogits_;
};

}  // namespace titan::nn
