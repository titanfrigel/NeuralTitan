#include "ops/cpu/layout.hpp"

#include <algorithm>
#include <vector>

namespace titan::ops::cpu {

Tensor swap_dims_1_2(const Tensor& x) {
    const std::int64_t n_a = x.dim(0);
    const std::int64_t n_b = x.dim(1);
    const std::int64_t n_c = x.dim(2);
    const std::int64_t n_d = x.dim(3);
    Tensor y = Tensor::empty({n_a, n_c, n_b, n_d}, x.device());
    const float* px = x.data();
    float* py = y.data();
    // Each innermost row of n_d values moves as a block:
    // x[a, b, c, :] -> y[a, c, b, :].
    for (std::int64_t a = 0; a < n_a; ++a) {
        for (std::int64_t b = 0; b < n_b; ++b) {
            for (std::int64_t c = 0; c < n_c; ++c) {
                const float* src = px + ((a * n_b + b) * n_c + c) * n_d;
                float* dst = py + ((a * n_c + c) * n_b + b) * n_d;
                std::copy(src, src + n_d, dst);
            }
        }
    }
    return y;
}

Tensor repeat_kv(const Tensor& x, std::int64_t n_rep) {
    const std::int64_t batch = x.dim(0);
    const std::int64_t kv_heads = x.dim(1);
    // One head is a contiguous (T, hd) block.
    const std::int64_t block = x.dim(2) * x.dim(3);
    Tensor y = Tensor::empty({batch, kv_heads * n_rep, x.dim(2), x.dim(3)},
                             x.device());
    const float* px = x.data();
    float* py = y.data();
    // Output head h = g·n_rep + r copies KV head g = h / n_rep.
    for (std::int64_t b = 0; b < batch; ++b) {
        for (std::int64_t g = 0; g < kv_heads; ++g) {
            const float* src = px + (b * kv_heads + g) * block;
            for (std::int64_t r = 0; r < n_rep; ++r) {
                float* dst = py + ((b * kv_heads + g) * n_rep + r) * block;
                std::copy(src, src + block, dst);
            }
        }
    }
    return y;
}

Tensor repeat_kv_backward(const Tensor& dy, std::int64_t n_rep) {
    const std::int64_t batch = dy.dim(0);
    const std::int64_t kv_heads = dy.dim(1) / n_rep;
    const std::int64_t block = dy.dim(2) * dy.dim(3);
    Tensor dx =
        Tensor::empty({batch, kv_heads, dy.dim(2), dy.dim(3)}, dy.device());
    const float* pdy = dy.data();
    float* pdx = dx.data();
    // dx[b, g] = Σ_r dy[b, g·n_rep + r], summed in double.
    std::vector<double> acc(static_cast<std::size_t>(block));
    for (std::int64_t b = 0; b < batch; ++b) {
        for (std::int64_t g = 0; g < kv_heads; ++g) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (std::int64_t r = 0; r < n_rep; ++r) {
                const float* src =
                    pdy + ((b * kv_heads + g) * n_rep + r) * block;
                for (std::int64_t i = 0; i < block; ++i) {
                    acc[static_cast<std::size_t>(i)] += src[i];
                }
            }
            float* dst = pdx + (b * kv_heads + g) * block;
            for (std::int64_t i = 0; i < block; ++i) {
                dst[i] = static_cast<float>(acc[static_cast<std::size_t>(i)]);
            }
        }
    }
    return dx;
}

void kv_write(Tensor& cache, const Tensor& x, std::int64_t pos) {
    const std::int64_t bh = x.dim(0) * x.dim(1);
    const std::int64_t t_max = cache.dim(2);
    const std::int64_t t = x.dim(2);
    const std::int64_t hd = x.dim(3);
    const float* px = x.data();
    float* pc = cache.data();
    // Per (batch, head) the T new rows are one contiguous block of T·hd
    // values, and so is their destination inside the cache.
    for (std::int64_t i = 0; i < bh; ++i) {
        std::copy(px + i * t * hd, px + (i + 1) * t * hd,
                  pc + (i * t_max + pos) * hd);
    }
}

Tensor kv_read(const Tensor& cache, std::int64_t len) {
    const std::int64_t bh = cache.dim(0) * cache.dim(1);
    const std::int64_t t_max = cache.dim(2);
    const std::int64_t hd = cache.dim(3);
    Tensor y =
        Tensor::empty({cache.dim(0), cache.dim(1), len, hd}, cache.device());
    const float* pc = cache.data();
    float* py = y.data();
    // The first len rows of each (batch, head) are one contiguous block.
    for (std::int64_t i = 0; i < bh; ++i) {
        std::copy(pc + i * t_max * hd, pc + (i * t_max + len) * hd,
                  py + i * len * hd);
    }
    return y;
}

}  // namespace titan::ops::cpu
