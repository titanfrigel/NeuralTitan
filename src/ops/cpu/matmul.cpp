#include "ops/cpu/matmul.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/parallel.hpp"

namespace titan::ops::cpu {

namespace {

// Number of k terms summed in float before being added to the double total.
constexpr std::int64_t kBlock = 64;

struct Dims {
    std::int64_t batch, m, k, n;
};

// Reads the matrix sizes off the tensors. Leading dimensions are batch.
Dims dims_of(const Tensor& a, const Tensor& b, bool ta, bool tb) {
    const std::int64_t a0 = a.dim(-2), a1 = a.dim(-1);
    Dims d;
    d.m = ta ? a1 : a0;
    d.k = ta ? a0 : a1;
    d.n = tb ? b.dim(-2) : b.dim(-1);
    d.batch = a.numel() / (a0 * a1);
    return d;
}

}  // namespace

void matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a,
                bool transpose_b) {
    const Dims d = dims_of(a, b, transpose_a, transpose_b);
    const std::int64_t mk = d.m * d.k, kn = d.k * d.n, mn = d.m * d.n;

    // Pack op(b) as a contiguous (K, N) matrix per batch so that the inner
    // loop below always streams a row of it. When b is not transposed it is
    // already in that layout and is used in place.
    std::vector<float> packed;
    const float* pb = b.data();
    if (transpose_b) {
        packed.resize(static_cast<std::size_t>(d.batch * kn));
        for (std::int64_t t = 0; t < d.batch; ++t) {
            const float* src = b.data() + t * kn;  // stored (N, K)
            float* dst = packed.data() + t * kn;   // wanted (K, N)
            for (std::int64_t j = 0; j < d.n; ++j) {
                for (std::int64_t p = 0; p < d.k; ++p) {
                    dst[p * d.n + j] = src[j * d.k + p];
                }
            }
        }
        pb = packed.data();
    }
    const float* pa = a.data();
    float* po = out.data();

    parallel_for(
        0, d.batch * d.m, [&](std::int64_t row_begin, std::int64_t row_end) {
            std::vector<float> block(static_cast<std::size_t>(d.n));
            std::vector<double> total(static_cast<std::size_t>(d.n));
            for (std::int64_t row = row_begin; row < row_end; ++row) {
                const std::int64_t t = row / d.m;  // batch index
                const std::int64_t i = row % d.m;  // row of op(a)
                const float* a_mat = pa + t * mk;
                const float* b_mat = pb + t * kn;
                float* c_row = po + t * mn + i * d.n;
                std::fill(total.begin(), total.end(), 0.0);
                for (std::int64_t k0 = 0; k0 < d.k; k0 += kBlock) {
                    const std::int64_t k1 = std::min(k0 + kBlock, d.k);
                    std::fill(block.begin(), block.end(), 0.0f);
                    for (std::int64_t p = k0; p < k1; ++p) {
                        // op(a)[i][p]; a is stored (K, M) when transposed.
                        const float a_ip = transpose_a ? a_mat[p * d.m + i]
                                                       : a_mat[i * d.k + p];
                        const float* b_row = b_mat + p * d.n;
                        float* __restrict acc = block.data();
                        for (std::int64_t j = 0; j < d.n; ++j) {
                            acc[j] += a_ip * b_row[j];
                        }
                    }
                    for (std::int64_t j = 0; j < d.n; ++j) {
                        total[static_cast<std::size_t>(j)] +=
                            block[static_cast<std::size_t>(j)];
                    }
                }
                for (std::int64_t j = 0; j < d.n; ++j) {
                    c_row[j] +=
                        static_cast<float>(total[static_cast<std::size_t>(j)]);
                }
            }
        });
}

void matmul_naive(Tensor& out, const Tensor& a, const Tensor& b,
                  bool transpose_a, bool transpose_b) {
    const Dims d = dims_of(a, b, transpose_a, transpose_b);
    const float* pa = a.data();
    const float* pb = b.data();
    float* po = out.data();
    for (std::int64_t t = 0; t < d.batch; ++t) {
        const float* a_mat = pa + t * d.m * d.k;
        const float* b_mat = pb + t * d.k * d.n;
        float* c_mat = po + t * d.m * d.n;
        for (std::int64_t i = 0; i < d.m; ++i) {
            for (std::int64_t j = 0; j < d.n; ++j) {
                double acc = 0.0;
                for (std::int64_t p = 0; p < d.k; ++p) {
                    const float x =
                        transpose_a ? a_mat[p * d.m + i] : a_mat[i * d.k + p];
                    const float y =
                        transpose_b ? b_mat[j * d.k + p] : b_mat[p * d.n + j];
                    acc += static_cast<double>(x) * y;
                }
                c_mat[i * d.n + j] = static_cast<float>(acc);
            }
        }
    }
}

}  // namespace titan::ops::cpu
