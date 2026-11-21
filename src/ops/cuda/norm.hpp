#pragma once

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA implementations of ops/norm.hpp, on CUDA tensors. They do not validate;
// the dispatching functions in ops/norm.cpp do. They allocate each saved
// statistic whose output pointer is non-null, launch on the default stream and
// return the device result without waiting. The backward functions add the
// parameter gradients into the given accumulators with `+=`.

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps,
                Tensor* inv_rms_out);
Tensor rms_norm_backward(const Tensor& x, const Tensor& weight,
                         const Tensor& inv_rms, const Tensor& dy,
                         Tensor& dweight_acc);
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  float eps, Tensor* mean_out, Tensor* inv_std_out);
Tensor layer_norm_backward(const Tensor& x, const Tensor& weight,
                           const Tensor& mean, const Tensor& inv_std,
                           const Tensor& dy, Tensor& dweight_acc,
                           Tensor& dbias_acc);

}  // namespace titan::ops::cuda
