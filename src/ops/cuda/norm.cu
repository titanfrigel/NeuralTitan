#include "ops/cuda/norm.hpp"

#include <cstdint>

#include "ops/cuda/common.cuh"

namespace titan::ops::cuda {

namespace {

// Shape of a per-row statistic, as on the CPU: the shape of x without its last
// dimension, or {1} for a rank-1 x.
Shape stat_shape(const Tensor& x) {
    if (x.rank() == 1) return {1};
    return Shape(x.shape().begin(), x.shape().end() - 1);
}

// One block per row, rows on gridDim.x.
unsigned int row_grid(std::int64_t rows) {
    return static_cast<unsigned int>(rows);
}

// y = x * inv_rms * w. Thread t sums x_i^2 over i = t, t + 256, ... in float,
// then the shared block tree combines the 256 partials. inv_rms_out may be
// null.
__global__ void rms_norm_kernel(const float* x, const float* w, float* y,
                                float* inv_rms_out, std::int64_t d, float eps) {
    __shared__ float shared[kBlock];
    const std::int64_t r = blockIdx.x;
    const float* px = x + r * d;
    float* py = y + r * d;
    float ss = 0.0f;
    for (std::int64_t i = threadIdx.x; i < d; i += kBlock) ss += px[i] * px[i];
    ss = block_reduce(ss, shared, SumOp());
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(d) + eps);
    for (std::int64_t i = threadIdx.x; i < d; i += kBlock) {
        py[i] = px[i] * inv * w[i];
    }
    if (inv_rms_out != nullptr && threadIdx.x == 0) inv_rms_out[r] = inv;
}

// dx_i = r * g_i - x_i * r^3 * dot / D with g = dy * w and dot = sum g_i x_i.
__global__ void rms_norm_dx_kernel(const float* x, const float* w,
                                   const float* inv_rms, const float* dy,
                                   float* dx, std::int64_t d) {
    __shared__ float shared[kBlock];
    const std::int64_t r = blockIdx.x;
    const float* px = x + r * d;
    const float* pd = dy + r * d;
    float* pdx = dx + r * d;
    const float inv = inv_rms[r];
    float dot = 0.0f;
    for (std::int64_t i = threadIdx.x; i < d; i += kBlock) {
        dot += pd[i] * w[i] * px[i];
    }
    dot = block_reduce(dot, shared, SumOp());
    const float proj = inv * inv * inv * dot / static_cast<float>(d);
    for (std::int64_t i = threadIdx.x; i < d; i += kBlock) {
        pdx[i] = inv * pd[i] * w[i] - px[i] * proj;
    }
}

// dweight_i += sum_r dy[r, i] * x[r, i] * inv_rms[r]. One thread per column;
// the rows are summed in order in double, then added once to the accumulator.
// Neighbouring threads read neighbouring columns, so the loads coalesce.
__global__ void rms_norm_dweight_kernel(const float* x, const float* inv_rms,
                                        const float* dy, float* dweight,
                                        std::int64_t rows, std::int64_t d) {
    for (std::int64_t i = global_thread_index(); i < d; i += grid_stride()) {
        double acc = 0.0;
        for (std::int64_t r = 0; r < rows; ++r) {
            acc +=
                static_cast<double>(dy[r * d + i]) * x[r * d + i] * inv_rms[r];
        }
        dweight[i] += static_cast<float>(acc);
    }
}

// y = (x - mu) * s * w + b. Two block reductions per row: the mean, then the
// variance of the centred values (as on the CPU). The statistic outputs may be
// null.
__global__ void layer_norm_kernel(const float* x, const float* w,
                                  const float* b, float* y, float* mean_out,
                                  float* inv_std_out, std::int64_t d,
                                  float eps) {
    __shared__ float shared[kBlock];
    const std::int64_t r = blockIdx.x;
    const float* px = x + r * d;
    float* py = y + r * d;
    float sum = 0.0f;
    for (std::int64_t i = threadIdx.x; i < d; i += kBlock) sum += px[i];
    sum = block_reduce(sum, shared, SumOp());
    const float mu = sum / static_cast<float>(d);
    float sq = 0.0f;
    for (std::int64_t i = threadIdx.x; i < d; i += kBlock) {
        const float c = px[i] - mu;
        sq += c * c;
    }
    sq = block_reduce(sq, shared, SumOp());
    const float inv = 1.0f / sqrtf(sq / static_cast<float>(d) + eps);
    for (std::int64_t i = threadIdx.x; i < d; i += kBlock) {
        py[i] = (px[i] - mu) * inv * w[i] + b[i];
    }
    if (threadIdx.x == 0) {
        if (mean_out != nullptr) mean_out[r] = mu;
        if (inv_std_out != nullptr) inv_std_out[r] = inv;
    }
}

// dx_i = s * (g_i - mean(g) - xhat_i * mean(g * xhat)), g = dy * w.
__global__ void layer_norm_dx_kernel(const float* x, const float* w,
                                     const float* mean, const float* inv_std,
                                     const float* dy, float* dx,
                                     std::int64_t d) {
    __shared__ float shared[kBlock];
    const std::int64_t r = blockIdx.x;
    const float* px = x + r * d;
    const float* pd = dy + r * d;
    float* pdx = dx + r * d;
    const float mu = mean[r];
    const float inv = inv_std[r];
    float sum_g = 0.0f;
    float sum_gx = 0.0f;
    for (std::int64_t i = threadIdx.x; i < d; i += kBlock) {
        const float xhat = (px[i] - mu) * inv;
        const float g = pd[i] * w[i];
        sum_g += g;
        sum_gx += g * xhat;
    }
    sum_g = block_reduce(sum_g, shared, SumOp());
    sum_gx = block_reduce(sum_gx, shared, SumOp());
    const float inv_d = 1.0f / static_cast<float>(d);
    for (std::int64_t i = threadIdx.x; i < d; i += kBlock) {
        const float xhat = (px[i] - mu) * inv;
        const float g = pd[i] * w[i];
        pdx[i] = inv * (g - sum_g * inv_d - xhat * sum_gx * inv_d);
    }
}

// dweight_i += sum_r dy * xhat, dbias_i += sum_r dy; the same mapping as the
// RMSNorm column kernel.
__global__ void layer_norm_dparams_kernel(const float* x, const float* mean,
                                          const float* inv_std, const float* dy,
                                          float* dweight, float* dbias,
                                          std::int64_t rows, std::int64_t d) {
    for (std::int64_t i = global_thread_index(); i < d; i += grid_stride()) {
        double dw = 0.0;
        double db = 0.0;
        for (std::int64_t r = 0; r < rows; ++r) {
            const double g = dy[r * d + i];
            const double xhat =
                (static_cast<double>(x[r * d + i]) - mean[r]) * inv_std[r];
            dw += g * xhat;
            db += g;
        }
        dweight[i] += static_cast<float>(dw);
        dbias[i] += static_cast<float>(db);
    }
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
    rms_norm_kernel<<<row_grid(rows), kBlock>>>(
        x.data(), weight.data(), y.data(),
        inv_rms_out != nullptr ? inv_rms_out->data() : nullptr, d, eps);
    check_launch();
    return y;
}

Tensor rms_norm_backward(const Tensor& x, const Tensor& weight,
                         const Tensor& inv_rms, const Tensor& dy,
                         Tensor& dweight_acc) {
    const std::int64_t d = x.dim(-1);
    const std::int64_t rows = x.numel() / d;
    Tensor dx = Tensor::empty(x.shape(), x.device());
    rms_norm_dx_kernel<<<row_grid(rows), kBlock>>>(
        x.data(), weight.data(), inv_rms.data(), dy.data(), dx.data(), d);
    check_launch();
    rms_norm_dweight_kernel<<<grid_for(d, kBlock), kBlock>>>(
        x.data(), inv_rms.data(), dy.data(), dweight_acc.data(), rows, d);
    check_launch();
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
    layer_norm_kernel<<<row_grid(rows), kBlock>>>(
        x.data(), weight.data(), bias.data(), y.data(),
        mean_out != nullptr ? mean_out->data() : nullptr,
        inv_std_out != nullptr ? inv_std_out->data() : nullptr, d, eps);
    check_launch();
    return y;
}

Tensor layer_norm_backward(const Tensor& x, const Tensor& weight,
                           const Tensor& mean, const Tensor& inv_std,
                           const Tensor& dy, Tensor& dweight_acc,
                           Tensor& dbias_acc) {
    const std::int64_t d = x.dim(-1);
    const std::int64_t rows = x.numel() / d;
    Tensor dx = Tensor::empty(x.shape(), x.device());
    layer_norm_dx_kernel<<<row_grid(rows), kBlock>>>(
        x.data(), weight.data(), mean.data(), inv_std.data(), dy.data(),
        dx.data(), d);
    check_launch();
    layer_norm_dparams_kernel<<<grid_for(d, kBlock), kBlock>>>(
        x.data(), mean.data(), inv_std.data(), dy.data(), dweight_acc.data(),
        dbias_acc.data(), rows, d);
    check_launch();
    return dx;
}

}  // namespace titan::ops::cuda
