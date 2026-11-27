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
// The narrowest column tile, so the inner loop still vectorizes.
constexpr std::int64_t kMinTile = 16;
// With fewer rows of op(a) than this per batch, a transposed b is read in
// place instead of packed: packing costs as much as one row's products.
constexpr std::int64_t kPackMinRows = 4;
// Work items wanted per thread, so uneven items still balance.
constexpr std::int64_t kItemsPerThread = 4;

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

// The width of one column tile: the whole row when there are enough rows to
// keep every thread busy, narrower (but at least kMinTile) when there are
// few, as in decoding with M = 1.
std::int64_t tile_width(std::int64_t rows, std::int64_t n) {
    const std::int64_t wanted = kItemsPerThread * num_threads();
    const std::int64_t max_tiles = (n + kMinTile - 1) / kMinTile;
    const std::int64_t tiles =
        std::clamp((wanted + rows - 1) / rows, std::int64_t{1}, max_tiles);
    return (n + tiles - 1) / tiles;
}

}  // namespace

void matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a,
                bool transpose_b) {
    const Dims d = dims_of(a, b, transpose_a, transpose_b);
    const std::int64_t mk = d.m * d.k, kn = d.k * d.n, mn = d.m * d.n;

    // With many rows, pack op(b) as a contiguous (K, N) matrix per batch so
    // that the inner loop streams a row of it; the packing is split over the
    // threads by (batch, column). With few rows a transposed b is read in
    // place: row j of b is column j of op(b), contiguous in k.
    const bool in_place = transpose_b && d.m < kPackMinRows;
    std::vector<float> packed;
    const float* pb = b.data();
    if (transpose_b && !in_place) {
        packed.resize(static_cast<std::size_t>(d.batch * kn));
        const float* src_all = b.data();
        float* dst_all = packed.data();
        parallel_for(
            0, d.batch * d.n, [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t col = begin; col < end; ++col) {
                    const std::int64_t t = col / d.n, j = col % d.n;
                    const float* src =
                        src_all + t * kn + j * d.k;     // stored (N, K)
                    float* dst = dst_all + t * kn + j;  // wanted (K, N)
                    for (std::int64_t p = 0; p < d.k; ++p)
                        dst[p * d.n] = src[p];
                }
            });
        pb = packed.data();
    }
    const float* pa = a.data();
    float* po = out.data();

    // One work item is one output row (t, i) restricted to the columns of
    // one tile. Every output element is computed by exactly one item with
    // the same accumulation order as matmul_acc_reference, so the result
    // does not depend on the tiling or the thread count.
    const std::int64_t width = tile_width(d.batch * d.m, d.n);
    const std::int64_t tiles = (d.n + width - 1) / width;
    parallel_for(
        0, d.batch * d.m * tiles,
        [&](std::int64_t item_begin, std::int64_t item_end) {
            std::vector<float> block(static_cast<std::size_t>(width));
            std::vector<double> total(static_cast<std::size_t>(width));
            for (std::int64_t item = item_begin; item < item_end; ++item) {
                const std::int64_t row = item / tiles;
                const std::int64_t j0 = (item % tiles) * width;
                const std::int64_t w = std::min(width, d.n - j0);
                const std::int64_t t = row / d.m;  // batch index
                const std::int64_t i = row % d.m;  // row of op(a)
                const float* a_mat = pa + t * mk;
                float* c_row = po + t * mn + i * d.n + j0;
                // op(a)[i][p]; a is stored (K, M) when transposed.
                auto a_at = [&](std::int64_t p) {
                    return transpose_a ? a_mat[p * d.m + i]
                                       : a_mat[i * d.k + p];
                };

                if (in_place) {
                    // A dot product per column: the k terms of one element in
                    // the same order as below, read along row j0 + j of b.
                    for (std::int64_t j = 0; j < w; ++j) {
                        const float* b_row = b.data() + t * kn + (j0 + j) * d.k;
                        double sum = 0.0;
                        for (std::int64_t k0 = 0; k0 < d.k; k0 += kBlock) {
                            const std::int64_t k1 = std::min(k0 + kBlock, d.k);
                            float part = 0.0f;
                            for (std::int64_t p = k0; p < k1; ++p) {
                                part += a_at(p) * b_row[p];
                            }
                            sum += part;
                        }
                        c_row[j] += static_cast<float>(sum);
                    }
                    continue;
                }

                // The i-k-j order over the tile: the innermost loop adds
                // op(a)[i][p] times a contiguous piece of row p of op(b).
                const float* b_mat = pb + t * kn + j0;
                std::fill(total.begin(), total.begin() + w, 0.0);
                for (std::int64_t k0 = 0; k0 < d.k; k0 += kBlock) {
                    const std::int64_t k1 = std::min(k0 + kBlock, d.k);
                    std::fill(block.begin(), block.begin() + w, 0.0f);
                    for (std::int64_t p = k0; p < k1; ++p) {
                        const float a_ip = a_at(p);
                        const float* b_row = b_mat + p * d.n;
                        float* __restrict acc = block.data();
                        for (std::int64_t j = 0; j < w; ++j) {
                            acc[j] += a_ip * b_row[j];
                        }
                    }
                    for (std::int64_t j = 0; j < w; ++j) {
                        total[static_cast<std::size_t>(j)] +=
                            block[static_cast<std::size_t>(j)];
                    }
                }
                for (std::int64_t j = 0; j < w; ++j) {
                    c_row[j] +=
                        static_cast<float>(total[static_cast<std::size_t>(j)]);
                }
            }
        });
}

void matmul_acc_reference(Tensor& out, const Tensor& a, const Tensor& b,
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
                double sum = 0.0;
                for (std::int64_t k0 = 0; k0 < d.k; k0 += kBlock) {
                    const std::int64_t k1 = std::min(k0 + kBlock, d.k);
                    float part = 0.0f;
                    for (std::int64_t p = k0; p < k1; ++p) {
                        const float x = transpose_a ? a_mat[p * d.m + i]
                                                    : a_mat[i * d.k + p];
                        const float y = transpose_b ? b_mat[j * d.k + p]
                                                    : b_mat[p * d.n + j];
                        part += x * y;
                    }
                    sum += part;
                }
                c_mat[i * d.n + j] += static_cast<float>(sum);
            }
        }
    }
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
