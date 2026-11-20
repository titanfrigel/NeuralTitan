#include "ops/cpu/norm.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace titan::ops::cpu {

namespace {

// Shape of a per-row statistic: x's shape without the last dimension, or {1}
// for a rank-1 x.
Shape stat_shape(const Tensor& x) {
    if (x.rank() == 1) return {1};
    return Shape(x.shape().begin(), x.shape().end() - 1);
}

}  // namespace

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps,
                Tensor* inv_rms_out) {
    const std::int64_t d = x.dim(-1);
    const std::int64_t rows = x.numel() / d;
    Tensor y = Tensor::empty(x.shape(), x.device());
    if (inv_rms_out != nullptr) {
        *inv_rms_out = Tensor::empty(stat_shape(x), x.device());
    }
    const float* w = weight.data();
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* px = x.data() + r * d;
        float* py = y.data() + r * d;
        double ss = 0.0;
        for (std::int64_t i = 0; i < d; ++i) {
            ss += static_cast<double>(px[i]) * px[i];
        }
        const float inv =
            static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(d) +
                                               static_cast<double>(eps)));
        for (std::int64_t i = 0; i < d; ++i) py[i] = px[i] * inv * w[i];
        if (inv_rms_out != nullptr) inv_rms_out->data()[r] = inv;
    }
    return y;
}

Tensor rms_norm_backward(const Tensor& x, const Tensor& weight,
                         const Tensor& inv_rms, const Tensor& dy,
                         Tensor& dweight_acc) {
    const std::int64_t d = x.dim(-1);
    const std::int64_t rows = x.numel() / d;
    Tensor dx = Tensor::empty(x.shape(), x.device());
    const float* w = weight.data();
    std::vector<double> dw(static_cast<std::size_t>(d), 0.0);
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* px = x.data() + r * d;
        const float* pd = dy.data() + r * d;
        float* pdx = dx.data() + r * d;
        const double inv = inv_rms.data()[r];
        // g_i = dy_i * w_i; dot = sum_i g_i * x_i.
        double dot = 0.0;
        for (std::int64_t i = 0; i < d; ++i) {
            dot += static_cast<double>(pd[i]) * w[i] * px[i];
        }
        // dx_i = r * g_i - x_i * r^3 * dot / D.
        const double proj = inv * inv * inv * dot / static_cast<double>(d);
        for (std::int64_t i = 0; i < d; ++i) {
            pdx[i] = static_cast<float>(inv * pd[i] * w[i] - px[i] * proj);
            dw[static_cast<std::size_t>(i)] +=
                static_cast<double>(pd[i]) * px[i] * inv;
        }
    }
    float* dw_out = dweight_acc.data();
    for (std::int64_t i = 0; i < d; ++i) {
        dw_out[i] += static_cast<float>(dw[static_cast<std::size_t>(i)]);
    }
    return dx;
}

Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  float eps, Tensor* mean_out, Tensor* inv_std_out) {
    const std::int64_t d = x.dim(-1);
    const std::int64_t rows = x.numel() / d;
    Tensor y = Tensor::empty(x.shape(), x.device());
    if (mean_out != nullptr) {
        *mean_out = Tensor::empty(stat_shape(x), x.device());
    }
    if (inv_std_out != nullptr) {
        *inv_std_out = Tensor::empty(stat_shape(x), x.device());
    }
    const float* w = weight.data();
    const float* b = bias.data();
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* px = x.data() + r * d;
        float* py = y.data() + r * d;
        double sum = 0.0;
        for (std::int64_t i = 0; i < d; ++i) sum += px[i];
        const double mu = sum / static_cast<double>(d);
        double sq = 0.0;
        for (std::int64_t i = 0; i < d; ++i) {
            const double c = px[i] - mu;
            sq += c * c;
        }
        const double var = sq / static_cast<double>(d);
        const double inv = 1.0 / std::sqrt(var + static_cast<double>(eps));
        for (std::int64_t i = 0; i < d; ++i) {
            py[i] = static_cast<float>((px[i] - mu) * inv * w[i] + b[i]);
        }
        if (mean_out != nullptr) {
            mean_out->data()[r] = static_cast<float>(mu);
        }
        if (inv_std_out != nullptr) {
            inv_std_out->data()[r] = static_cast<float>(inv);
        }
    }
    return y;
}

Tensor layer_norm_backward(const Tensor& x, const Tensor& weight,
                           const Tensor& mean, const Tensor& inv_std,
                           const Tensor& dy, Tensor& dweight_acc,
                           Tensor& dbias_acc) {
    const std::int64_t d = x.dim(-1);
    const std::int64_t rows = x.numel() / d;
    Tensor dx = Tensor::empty(x.shape(), x.device());
    const float* w = weight.data();
    std::vector<double> dw(static_cast<std::size_t>(d), 0.0);
    std::vector<double> db(static_cast<std::size_t>(d), 0.0);
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* px = x.data() + r * d;
        const float* pd = dy.data() + r * d;
        float* pdx = dx.data() + r * d;
        const double mu = mean.data()[r];
        const double inv = inv_std.data()[r];
        // xhat_i = (x_i - mu) * s; g_i = dy_i * w_i.
        double sum_g = 0.0;
        double sum_gx = 0.0;
        for (std::int64_t i = 0; i < d; ++i) {
            const double xhat = (px[i] - mu) * inv;
            const double g = static_cast<double>(pd[i]) * w[i];
            sum_g += g;
            sum_gx += g * xhat;
        }
        // dx_i = s * (g_i - mean(g) - xhat_i * mean(g * xhat)).
        const double inv_d = 1.0 / static_cast<double>(d);
        for (std::int64_t i = 0; i < d; ++i) {
            const double xhat = (px[i] - mu) * inv;
            const double g = static_cast<double>(pd[i]) * w[i];
            pdx[i] = static_cast<float>(
                inv * (g - sum_g * inv_d - xhat * sum_gx * inv_d));
            dw[static_cast<std::size_t>(i)] += pd[i] * xhat;
            db[static_cast<std::size_t>(i)] += pd[i];
        }
    }
    // One float add per parameter keeps += semantics; the row sum was double.
    float* dw_out = dweight_acc.data();
    float* db_out = dbias_acc.data();
    for (std::int64_t i = 0; i < d; ++i) {
        dw_out[i] += static_cast<float>(dw[static_cast<std::size_t>(i)]);
        db_out[i] += static_cast<float>(db[static_cast<std::size_t>(i)]);
    }
    return dx;
}

}  // namespace titan::ops::cpu
