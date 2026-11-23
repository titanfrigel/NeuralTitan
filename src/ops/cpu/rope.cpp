#include "ops/cpu/rope.hpp"

#include <cmath>

namespace titan::ops::cpu {

namespace {

// Rotates every pair (i, i + hd/2) of every head vector by sign * a, where
// a = p * theta^(-2i/hd). sign = +1 is the forward rotation; sign = -1 is its
// inverse, which is also its transpose. Angles and their cos/sin are computed
// in double and applied to float values.
Tensor rotate(const Tensor& x, std::int64_t pos_offset, float theta,
              double sign) {
    const std::int64_t t_len = x.dim(2);
    const std::int64_t hd = x.dim(3);
    const std::int64_t half = hd / 2;
    const std::int64_t heads = x.dim(0) * x.dim(1);

    Tensor y = Tensor::empty(x.shape(), x.device());
    const float* px = x.data();
    float* py = y.data();
    for (std::int64_t h = 0; h < heads; ++h) {
        for (std::int64_t t = 0; t < t_len; ++t) {
            const double p = static_cast<double>(pos_offset + t);
            const std::int64_t base = (h * t_len + t) * hd;
            for (std::int64_t i = 0; i < half; ++i) {
                const double a = p * std::pow(static_cast<double>(theta),
                                              -2.0 * static_cast<double>(i) /
                                                  static_cast<double>(hd));
                const float c = static_cast<float>(std::cos(a));
                const float s = static_cast<float>(sign * std::sin(a));
                const float lo = px[base + i];
                const float hi = px[base + i + half];
                py[base + i] = lo * c - hi * s;
                py[base + i + half] = hi * c + lo * s;
            }
        }
    }
    return y;
}

}  // namespace

Tensor rope(const Tensor& x, std::int64_t pos_offset, float theta) {
    return rotate(x, pos_offset, theta, 1.0);
}

Tensor rope_backward(const Tensor& dy, std::int64_t pos_offset, float theta) {
    return rotate(dy, pos_offset, theta, -1.0);
}

}  // namespace titan::ops::cpu
