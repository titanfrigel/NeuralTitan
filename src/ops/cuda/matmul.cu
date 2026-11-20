#include "ops/cuda/matmul.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "core/error.hpp"
#include "ops/cuda/common.cuh"
#include "ops/cuda/cublas_handle.cuh"

namespace titan::ops::cuda {

namespace {

// Edge of the square output tile, shared-memory tile and thread block of the
// tiled kernel: 16 x 16 = 256 threads, two 1 KiB tiles of shared memory.
constexpr int kTile = 16;

// The sizes of one product, read from the tensors on the host. op(a) is
// (m, k), op(b) is (k, n), out is (m, n), repeated `batch` times.
struct Gemm {
    std::int64_t batch;
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
    bool ta;
    bool tb;
};

Gemm gemm_of(const Tensor& out, const Tensor& a, bool ta, bool tb) {
    Gemm g{};
    g.m = out.dim(-2);
    g.n = out.dim(-1);
    g.k = ta ? a.dim(-2) : a.dim(-1);
    g.batch = out.numel() / (g.m * g.n);
    g.ta = ta;
    g.tb = tb;
    return g;
}

// op(a)[i][kk] of one row-major matrix: a is stored (m, k), or (k, m) when
// transposed (the table in docs/math/matmul.md).
__device__ inline float op_a(const float* a, const Gemm& g, std::int64_t i,
                             std::int64_t kk) {
    return g.ta ? a[kk * g.m + i] : a[i * g.k + kk];
}

// op(b)[kk][j] of one row-major matrix: b is stored (k, n), or (n, k) when
// transposed.
__device__ inline float op_b(const float* b, const Gemm& g, std::int64_t kk,
                             std::int64_t j) {
    return g.tb ? b[j * g.k + kk] : b[kk * g.n + j];
}

// One thread per output element, in a grid-stride loop over batch * m * n.
// j is the fastest index, so neighbouring threads write neighbouring outputs.
// The sum runs over k in increasing order in one float accumulator and is
// then added to out, so the result depends only on the shapes.
__global__ void matmul_naive_kernel(const float* a, const float* b, float* out,
                                    Gemm g) {
    const std::int64_t per_batch = g.m * g.n;
    const std::int64_t total = g.batch * per_batch;
    for (std::int64_t idx = global_thread_index(); idx < total;
         idx += grid_stride()) {
        const std::int64_t t = idx / per_batch;
        const std::int64_t r = idx % per_batch;
        const std::int64_t i = r / g.n;
        const std::int64_t j = r % g.n;
        const float* at = a + t * g.m * g.k;
        const float* bt = b + t * g.k * g.n;
        float acc = 0.0f;
        for (std::int64_t kk = 0; kk < g.k; ++kk) {
            acc += op_a(at, g, i, kk) * op_b(bt, g, kk, j);
        }
        out[idx] += acc;
    }
}

// One 16 x 16 thread block per 16 x 16 output tile. Every (batch, tile row,
// tile column) triple is folded into one flat tile index on gridDim.x, and the
// block loops over tile indices with the grid as its stride, so no grid
// dimension ever exceeds its limit. Thread (ty, tx) owns output (row, col).
//
// Per step of 16 along k, the block stages op(a)[row0.., k0..] and
// op(b)[k0.., col0..] in shared memory (every thread loads one element of
// each, zero outside the matrix), synchronizes, then each thread adds its 16
// products. A padded k contributes 0 * 0, so the sum is the naive kernel's
// sum in the same increasing k order.
__global__ void matmul_tiled_kernel(const float* a, const float* b, float* out,
                                    Gemm g, std::int64_t tiles_m,
                                    std::int64_t tiles_n) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const std::int64_t tiles_per_batch = tiles_m * tiles_n;
    const std::int64_t total_tiles = g.batch * tiles_per_batch;
    // The loop bounds depend only on blockIdx, so every thread of a block
    // takes the same iterations and reaches the same __syncthreads().
    for (std::int64_t tile = blockIdx.x; tile < total_tiles;
         tile += gridDim.x) {
        const std::int64_t t = tile / tiles_per_batch;
        const std::int64_t r = tile % tiles_per_batch;
        const std::int64_t row = (r / tiles_n) * kTile + ty;
        const std::int64_t col = (r % tiles_n) * kTile + tx;
        const float* at = a + t * g.m * g.k;
        const float* bt = b + t * g.k * g.n;
        float acc = 0.0f;
        for (std::int64_t k0 = 0; k0 < g.k; k0 += kTile) {
            const std::int64_t ka = k0 + tx;
            const std::int64_t kb = k0 + ty;
            a_tile[ty][tx] =
                (row < g.m && ka < g.k) ? op_a(at, g, row, ka) : 0.0f;
            b_tile[ty][tx] =
                (kb < g.k && col < g.n) ? op_b(bt, g, kb, col) : 0.0f;
            __syncthreads();
            for (int kk = 0; kk < kTile; ++kk) {
                acc += a_tile[ty][kk] * b_tile[kk][tx];
            }
            // The next step overwrites the tiles; wait until all have read.
            __syncthreads();
        }
        if (row < g.m && col < g.n) {
            out[t * g.m * g.n + row * g.n + col] += acc;
        }
    }
}

// cuBLAS takes its sizes, leading dimensions and batch count as int.
int as_cublas_int(std::int64_t value, const char* what) {
    TITAN_CHECK(value <= std::numeric_limits<int>::max(),
                std::string("ops::cuda::matmul: ") + what + " " +
                    std::to_string(value) + " exceeds cuBLAS's int range");
    return static_cast<int>(value);
}

// Row-major C (m x n) = op(A) op(B) through column-major cuBLAS. A row-major
// (R, C) matrix is, read column-major, the (C x R) matrix of its transpose
// with leading dimension C. So cuBLAS sees our C as C^T (n x m, ldc = n) and
// computes C^T = op(B)^T op(A)^T, B first:
// - stored b read column-major is op(b)^T if tb is false, op(b) if true, so
//   cuBLAS's first operand takes transpose tb, with ldb = b's last dimension;
// - likewise the second operand takes transpose ta, with lda = a's last
//   dimension.
// docs/math/matmul.md derives each case. beta = 1 accumulates into out.
void matmul_cublas(Tensor& out, const Tensor& a, const Tensor& b,
                   const Gemm& g) {
    const int m = as_cublas_int(g.n, "n");
    const int n = as_cublas_int(g.m, "m");
    const int k = as_cublas_int(g.k, "k");
    const int lda = as_cublas_int(a.dim(-1), "lda");
    const int ldb = as_cublas_int(b.dim(-1), "ldb");
    const int batch = as_cublas_int(g.batch, "batch count");
    const float alpha = 1.0f;
    const float beta = 1.0f;
    TITAN_CUBLAS_CHECK(cublasSgemmStridedBatched(
        cublas_handle(), g.tb ? CUBLAS_OP_T : CUBLAS_OP_N,
        g.ta ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, b.data(), ldb,
        g.k * g.n, a.data(), lda, g.m * g.k, &beta, out.data(), m, g.m * g.n,
        batch));
}

}  // namespace

void matmul_naive(Tensor& out, const Tensor& a, const Tensor& b,
                  bool transpose_a, bool transpose_b) {
    const Gemm g = gemm_of(out, a, transpose_a, transpose_b);
    matmul_naive_kernel<<<grid_for(out.numel(), kBlock), kBlock>>>(
        a.data(), b.data(), out.data(), g);
    check_launch();
}

void matmul_tiled(Tensor& out, const Tensor& a, const Tensor& b,
                  bool transpose_a, bool transpose_b) {
    const Gemm g = gemm_of(out, a, transpose_a, transpose_b);
    const std::int64_t tiles_m = (g.m + kTile - 1) / kTile;
    const std::int64_t tiles_n = (g.n + kTile - 1) / kTile;
    const std::int64_t total_tiles = g.batch * tiles_m * tiles_n;
    const auto blocks =
        static_cast<unsigned int>(std::min(total_tiles, kMaxBlocks));
    matmul_tiled_kernel<<<blocks, dim3(kTile, kTile)>>>(
        a.data(), b.data(), out.data(), g, tiles_m, tiles_n);
    check_launch();
}

void matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a,
                bool transpose_b, GemmBackend backend) {
    switch (backend) {
        case GemmBackend::naive:
            matmul_naive(out, a, b, transpose_a, transpose_b);
            return;
        case GemmBackend::tiled:
            matmul_tiled(out, a, b, transpose_a, transpose_b);
            return;
        case GemmBackend::cublas:
            matmul_cublas(out, a, b, gemm_of(out, a, transpose_a, transpose_b));
            return;
    }
}

}  // namespace titan::ops::cuda
