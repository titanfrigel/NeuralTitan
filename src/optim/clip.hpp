#pragma once

#include <vector>

#include "nn/module.hpp"

namespace titan::optim {

// Returns the global L2 norm of all gradients (accumulated in double) and,
// when it exceeds `max_norm`, scales every gradient by max_norm / norm. A
// non-finite norm is returned as is and nothing is scaled; the caller decides
// what to do.
double clip_grad_norm(const std::vector<nn::Parameter*>& params,
                      double max_norm);

}  // namespace titan::optim
