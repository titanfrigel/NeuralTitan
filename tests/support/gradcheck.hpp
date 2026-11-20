#pragma once

#include <functional>
#include <string>

#include "core/rng.hpp"
#include "core/tensor.hpp"

namespace titan::test {

// A CPU tensor of the given shape filled with scale * normal() draws.
Tensor random_tensor(Shape s, Rng& rng, float scale = 1.0f);

struct GradCheckResult {
    bool ok;
    // Empty when ok. Otherwise names the failing element index, the analytic
    // value and the numeric value.
    std::string detail;
};

// Checks `analytic_grad` (dL/d input) against central differences of
// L = sum(forward() * r), with r a fixed random tensor shaped like forward()'s
// output. `forward` must read `input`; the check perturbs it in place by
// +-1e-2 per element and restores it.
//
// An element passes when |analytic - numeric| <= 1e-3 + 2e-2 * |numeric|.
// At most 64 elements are checked: all of them when input has 64 or fewer,
// otherwise element floor(k * numel / 64) for k = 0..63, a fixed spread that
// needs no random draws.
GradCheckResult check_gradient(const std::function<Tensor()>& forward,
                               Tensor& input, const Tensor& analytic_grad,
                               const Tensor& r);

}  // namespace titan::test
