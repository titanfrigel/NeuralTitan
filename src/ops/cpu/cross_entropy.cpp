#include "ops/cpu/cross_entropy.hpp"

#include <cmath>

namespace titan::ops::cpu {

float cross_entropy(const Tensor& logits, std::span<const std::int32_t> targets,
                    Tensor* dlogits_out) {
    const std::int64_t n = logits.dim(0);
    const std::int64_t v = logits.dim(1);
    const float* pz = logits.data();
    float* pd = nullptr;
    if (dlogits_out != nullptr) {
        *dlogits_out = Tensor::empty({n, v}, logits.device());
        pd = dlogits_out->data();
    }
    const double inv_n = 1.0 / static_cast<double>(n);

    double total = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        const float* z = pz + i * v;
        // log-sum-exp with the row maximum removed: lse = m + log Σ exp(z - m).
        double m = z[0];
        for (std::int64_t j = 1; j < v; ++j) m = z[j] > m ? z[j] : m;
        double sum = 0.0;
        for (std::int64_t j = 0; j < v; ++j) sum += std::exp(z[j] - m);
        const double lse = m + std::log(sum);
        const std::int64_t t = targets[static_cast<std::size_t>(i)];
        total += lse - static_cast<double>(z[t]);
        if (pd != nullptr) {
            // (softmax - onehot) / N, with softmax_j = exp(z_j - lse).
            for (std::int64_t j = 0; j < v; ++j) {
                const double p = std::exp(z[j] - lse);
                pd[i * v + j] =
                    static_cast<float>((p - (j == t ? 1.0 : 0.0)) * inv_n);
            }
        }
    }
    return static_cast<float>(total * inv_n);
}

}  // namespace titan::ops::cpu
