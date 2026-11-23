#include "ops/cuda/rope.hpp"

#include <cstdint>

#include "ops/cuda/common.cuh"

namespace titan::ops::cuda {

namespace {

// One thread per rotation pair (row, i): row is a (batch, head, position)
// triple, i in [0, hd/2) names the pair of coordinates (i, i + hd/2). The
// angle a = p * theta^(-2i/hd) and its cos and sin are computed in double, as
// on the CPU, and rounded to float before they touch the data. sign = +1 is
// the forward rotation, sign = -1 the inverse (the backward pass).
__global__ void rope_kernel(const float* x, float* y, std::int64_t pairs,
                            std::int64_t t_len, std::int64_t hd,
                            std::int64_t pos_offset, double theta,
                            double sign) {
    const std::int64_t half = hd / 2;
    for (std::int64_t k = global_thread_index(); k < pairs;
         k += grid_stride()) {
        const std::int64_t row = k / half;
        const std::int64_t i = k % half;
        const std::int64_t t = row % t_len;
        const double p = static_cast<double>(pos_offset + t);
        const double a = p * pow(theta, -2.0 * static_cast<double>(i) /
                                            static_cast<double>(hd));
        const float c = static_cast<float>(cos(a));
        const float s = static_cast<float>(sign * sin(a));
        const std::int64_t lo_at = row * hd + i;
        const float lo = x[lo_at];
        const float hi = x[lo_at + half];
        y[lo_at] = lo * c - hi * s;
        y[lo_at + half] = hi * c + lo * s;
    }
}

Tensor rotate(const Tensor& x, std::int64_t pos_offset, float theta,
              double sign) {
    const std::int64_t pairs = x.numel() / 2;
    Tensor y = Tensor::empty(x.shape(), x.device());
    rope_kernel<<<grid_for(pairs, kBlock), kBlock>>>(
        x.data(), y.data(), pairs, x.dim(2), x.dim(3), pos_offset,
        static_cast<double>(theta), sign);
    check_launch();
    return y;
}

}  // namespace

Tensor rope(const Tensor& x, std::int64_t pos_offset, float theta) {
    return rotate(x, pos_offset, theta, 1.0);
}

Tensor rope_backward(const Tensor& dy, std::int64_t pos_offset, float theta) {
    return rotate(dy, pos_offset, theta, -1.0);
}

}  // namespace titan::ops::cuda
