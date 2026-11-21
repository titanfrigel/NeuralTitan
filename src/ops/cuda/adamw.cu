#include "ops/cuda/adamw.hpp"

#include <cmath>
#include <cstdint>

#include "ops/cuda/common.cuh"

namespace titan::ops::cuda {

namespace {

// One thread per element (grid-stride loop). The arithmetic is the CPU
// reference's, in double per element, so the two devices agree to the final
// rounding to float. The bias corrections c1 = 1 - beta1^step and
// c2 = 1 - beta2^step are computed once on the host and passed in.
__global__ void adamw_kernel(float* param, const float* grad, float* m,
                             float* v, std::int64_t n, double b1, double b2,
                             double c1, double c2, double lr, double eps,
                             double weight_decay) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        const double g = grad[i];
        const double mi = b1 * m[i] + (1.0 - b1) * g;
        const double vi = b2 * v[i] + (1.0 - b2) * g * g;
        m[i] = static_cast<float>(mi);
        v[i] = static_cast<float>(vi);
        const double m_hat = mi / c1;
        const double v_hat = vi / c2;
        const double p = param[i];
        // Decoupled weight decay: lambda * p sits outside the adaptive ratio.
        param[i] = static_cast<float>(
            p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p));
    }
}

}  // namespace

void adamw_update(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, std::int64_t step) {
    const double b1 = beta1;
    const double b2 = beta2;
    const double t = static_cast<double>(step);
    const double c1 = 1.0 - std::pow(b1, t);
    const double c2 = 1.0 - std::pow(b2, t);
    const std::int64_t n = param.numel();
    adamw_kernel<<<grid_for(n, kBlock), kBlock>>>(
        param.data(), grad.data(), m.data(), v.data(), n, b1, b2, c1, c2, lr,
        eps, weight_decay);
    check_launch();
}

}  // namespace titan::ops::cuda
