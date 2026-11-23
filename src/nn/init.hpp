#pragma once

#include "core/rng.hpp"
#include "nn/module.hpp"

namespace titan::nn {

// value ~ N(0, stddev^2), drawn from `rng`. The parameter must be on the CPU.
void init_normal(Parameter& p, Rng& rng, float stddev);
// Every element of value set to `value`. The parameter must be on the CPU.
void init_constant(Parameter& p, float value);

}  // namespace titan::nn
