#include "ops/cpu/adamw.hpp"

#include <cmath>

namespace titan::ops::cpu {

void adamw_update(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, std::int64_t step_count) {
    // Bias-correction terms in double: 1 - beta^step loses precision in float
    // when beta is close to 1.
    const double b1 = beta1;
    const double b2 = beta2;
    const double step = static_cast<double>(step_count);
    const double c1 = 1.0 - std::pow(b1, step);
    const double c2 = 1.0 - std::pow(b2, step);

    float* pp = param.data();
    const float* pg = grad.data();
    float* pm = m.data();
    float* pv = v.data();
    const std::int64_t n = param.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        const double g = pg[i];
        const double mi = b1 * pm[i] + (1.0 - b1) * g;
        const double vi = b2 * pv[i] + (1.0 - b2) * g * g;
        pm[i] = static_cast<float>(mi);
        pv[i] = static_cast<float>(vi);
        const double m_hat = mi / c1;
        const double v_hat = vi / c2;
        const double p = pp[i];
        pp[i] = static_cast<float>(
            p - lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p));
    }
}

}  // namespace titan::ops::cpu
