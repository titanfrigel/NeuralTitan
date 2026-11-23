#include "ops/cuda/cross_entropy.hpp"

#include <cstdint>

#include "ops/cuda/common.cuh"

namespace titan::ops::cuda {

namespace {

// One block of kBlock threads per row, rows on gridDim.x (a row count above
// 65535 would not fit gridDim.y). Thread t takes entries t, t + 256, ... of
// the row. Two fixed block trees give the row maximum m and the sum
// S = sum_j exp(z_j - m); then lse = m + log S and the row loss is
// lse - z_target, written to row_loss in double for the mean. When dlogits is
// non-null each thread also writes (exp(z_j - lse) - onehot_j) / N.
__global__ void cross_entropy_kernel(const float* logits,
                                     const std::int32_t* targets,
                                     std::int64_t v, float inv_n,
                                     double* row_loss, float* dlogits) {
    __shared__ float shared[kBlock];
    const std::int64_t r = blockIdx.x;
    const float* z = logits + r * v;

    float m = __int_as_float(0xff800000);  // -infinity
    for (std::int64_t j = threadIdx.x; j < v; j += kBlock) {
        if (z[j] > m) m = z[j];
    }
    m = block_reduce(m, shared, MaxOp());

    float s = 0.0f;
    for (std::int64_t j = threadIdx.x; j < v; j += kBlock) s += expf(z[j] - m);
    s = block_reduce(s, shared, SumOp());

    const float lse = m + logf(s);
    const std::int64_t t = targets[r];
    if (threadIdx.x == 0) {
        row_loss[r] = static_cast<double>(lse) - static_cast<double>(z[t]);
    }
    if (dlogits != nullptr) {
        float* g = dlogits + r * v;
        for (std::int64_t j = threadIdx.x; j < v; j += kBlock) {
            const float p = expf(z[j] - lse);
            g[j] = (p - (j == t ? 1.0f : 0.0f)) * inv_n;
        }
    }
}

// One block sums the n row losses in double. Thread t takes rows
// t, t + 256, ... in order, then the fixed block tree combines the threads,
// so the mean is the same on every run.
__global__ void sum_rows_kernel(const double* row_loss, std::int64_t n,
                                double* result) {
    __shared__ double shared[kBlock];
    double acc = 0.0;
    for (std::int64_t i = threadIdx.x; i < n; i += kBlock) acc += row_loss[i];
    const double total = block_reduce(acc, shared, SumOp());
    if (threadIdx.x == 0) *result = total;
}

}  // namespace

float cross_entropy(const Tensor& logits, std::span<const std::int32_t> targets,
                    Tensor* dlogits_out) {
    const std::int64_t n = logits.dim(0);
    const std::int64_t v = logits.dim(1);
    float* pd = nullptr;
    if (dlogits_out != nullptr) {
        *dlogits_out = Tensor::empty({n, v}, logits.device());
        pd = dlogits_out->data();
    }
    DeviceBuffer<std::int32_t> d_targets(n);
    TITAN_CUDA_CHECK(cudaMemcpy(d_targets.get(), targets.data(),
                                targets.size() * sizeof(std::int32_t),
                                cudaMemcpyHostToDevice));
    DeviceBuffer<double> row_loss(n);
    DeviceBuffer<double> total(1);
    const double inv_n = 1.0 / static_cast<double>(n);
    cross_entropy_kernel<<<static_cast<unsigned int>(n), kBlock>>>(
        logits.data(), d_targets.get(), v, static_cast<float>(inv_n),
        row_loss.get(), pd);
    check_launch();
    sum_rows_kernel<<<1, kBlock>>>(row_loss.get(), n, total.get());
    check_launch();
    double host = 0.0;
    // The copy on the default stream waits for both kernels.
    TITAN_CUDA_CHECK(
        cudaMemcpy(&host, total.get(), sizeof(double), cudaMemcpyDeviceToHost));
    return static_cast<float>(host * inv_n);
}

}  // namespace titan::ops::cuda
