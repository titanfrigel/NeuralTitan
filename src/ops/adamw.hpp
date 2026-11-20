#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace titan::ops {

// Hyperparameters of one AdamW step. `step` is the 1-based step count used
// for bias correction.
struct AdamWHyper {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    std::int64_t step;
};

// One AdamW update on CPU tensors, in place. param, grad, m and v have the same
// shape; h.step must be >= 1. With g = grad:
//   m = beta1 m + (1 - beta1) g
//   v = beta2 v + (1 - beta2) g^2
//   p -= lr (m/(1 - beta1^step) / (sqrt(v/(1 - beta2^step)) + eps)
//            + weight_decay p)
// A CUDA tensor throws "ops::adamw_update: CUDA implementation arrives in M2".
void adamw_update(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                  const AdamWHyper& h);

}  // namespace titan::ops
