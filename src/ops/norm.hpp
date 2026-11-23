#pragma once

#include "core/tensor.hpp"

namespace titan::ops {

// Normalization over the last dimension D of x. Row statistics accumulate in
// double. A CUDA tensor throws "ops::<name>: CUDA implementation arrives in
// M2". Saved statistics have the shape of x minus its last dimension, or {1}
// for a rank-1 x.

// y_i = x_i * r * w_i with r = 1 / sqrt(mean(x^2) + eps) per row. weight is
// {D}. When inv_rms_out is non-null it receives r.
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps,
                Tensor* inv_rms_out);
// Returns dx and adds dL/dweight into dweight_acc (shape {D}).
Tensor rms_norm_backward(const Tensor& x, const Tensor& weight,
                         const Tensor& inv_rms, const Tensor& dy,
                         Tensor& dweight_acc);

// y_i = (x_i - mu) * s * w_i + b_i with the biased variance and
// s = 1 / sqrt(var + eps) per row. weight and bias are {D}. When non-null,
// mean_out receives mu and inv_std_out receives s.
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  float eps, Tensor* mean_out, Tensor* inv_std_out);
// Returns dx and adds dL/dweight and dL/dbias into dweight_acc and dbias_acc
// (shape {D}).
Tensor layer_norm_backward(const Tensor& x, const Tensor& weight,
                           const Tensor& mean, const Tensor& inv_std,
                           const Tensor& dy, Tensor& dweight_acc,
                           Tensor& dbias_acc);

}  // namespace titan::ops
