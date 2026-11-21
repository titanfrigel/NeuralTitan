#include "ops/cuda/embedding.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ops/cuda/common.cuh"

namespace titan::ops::cuda {

namespace {

// One thread per output element, grid-stride over N * D. Element i belongs to
// lookup k = i / D, column j = i % D, and copies table[ids[k], j]. Neighbouring
// threads copy neighbouring columns of one row, so reads and writes coalesce.
__global__ void embedding_kernel(const float* table, const std::int32_t* ids,
                                 std::int64_t n, std::int64_t d, float* y) {
    for (std::int64_t i = global_thread_index(); i < n * d;
         i += grid_stride()) {
        const std::int64_t k = i / d;
        const std::int64_t j = i - k * d;
        y[i] = table[static_cast<std::int64_t>(ids[k]) * d + j];
    }
}

// Scatter-add without atomics. The ids are grouped on the host in CSR form:
// occurring id u is unique_ids[u], and its lookup positions are
// positions[offsets[u] .. offsets[u + 1]), in ascending order. One thread per
// (u, column), grid-stride over U * D: it sums dy[position, column] over those
// positions in order, in double, then adds the rounded sum to the accumulator
// once. No two threads write the same element, and the order is the CPU's.
__global__ void embedding_backward_kernel(const float* dy,
                                          const std::int32_t* unique_ids,
                                          const std::int64_t* offsets,
                                          const std::int64_t* positions,
                                          std::int64_t u_count, std::int64_t d,
                                          float* dtable) {
    for (std::int64_t i = global_thread_index(); i < u_count * d;
         i += grid_stride()) {
        const std::int64_t u = i / d;
        const std::int64_t j = i - u * d;
        double acc = 0.0;
        for (std::int64_t p = offsets[u]; p < offsets[u + 1]; ++p) {
            acc += dy[positions[p] * d + j];
        }
        dtable[static_cast<std::int64_t>(unique_ids[u]) * d + j] +=
            static_cast<float>(acc);
    }
}

// Copies a host array into a device buffer from the caching allocator.
template <typename T>
void upload(const DeviceBuffer<T>& dst, const T* src, std::size_t count) {
    TITAN_CUDA_CHECK(
        cudaMemcpy(dst.get(), src, count * sizeof(T), cudaMemcpyHostToDevice));
}

}  // namespace

Tensor embedding(const Tensor& table, std::span<const std::int32_t> ids,
                 Shape shape) {
    const std::int64_t d = table.dim(1);
    const auto n = static_cast<std::int64_t>(ids.size());
    Tensor y = Tensor::empty(std::move(shape), table.device());
    DeviceBuffer<std::int32_t> d_ids(n);
    upload(d_ids, ids.data(), ids.size());
    embedding_kernel<<<grid_for(n * d, kBlock), kBlock>>>(
        table.data(), d_ids.get(), n, d, y.data());
    check_launch();
    return y;
}

void embedding_backward(Tensor& dtable_acc, std::span<const std::int32_t> ids,
                        const Tensor& dy) {
    const std::int64_t vocab = dtable_acc.dim(0);
    const std::int64_t d = dtable_acc.dim(1);

    // Counting sort of the lookup positions by id. count[v] is the number of
    // lookups of id v; a prefix sum over the occurring ids gives each one's
    // start in `positions`. Scanning k in increasing order fills every id's
    // slice in ascending position order.
    std::vector<std::int64_t> count(static_cast<std::size_t>(vocab), 0);
    for (std::int32_t id : ids) ++count[static_cast<std::size_t>(id)];
    std::vector<std::int32_t> unique_ids;
    std::vector<std::int64_t> offsets = {0};
    std::vector<std::int64_t> next(static_cast<std::size_t>(vocab), 0);
    for (std::int64_t v = 0; v < vocab; ++v) {
        const std::int64_t c = count[static_cast<std::size_t>(v)];
        if (c == 0) continue;
        next[static_cast<std::size_t>(v)] = offsets.back();
        unique_ids.push_back(static_cast<std::int32_t>(v));
        offsets.push_back(offsets.back() + c);
    }
    std::vector<std::int64_t> positions(ids.size());
    for (std::size_t k = 0; k < ids.size(); ++k) {
        const auto v = static_cast<std::size_t>(ids[k]);
        positions[static_cast<std::size_t>(next[v]++)] =
            static_cast<std::int64_t>(k);
    }

    const auto u_count = static_cast<std::int64_t>(unique_ids.size());
    DeviceBuffer<std::int32_t> d_unique(u_count);
    DeviceBuffer<std::int64_t> d_offsets(u_count + 1);
    DeviceBuffer<std::int64_t> d_positions(
        static_cast<std::int64_t>(positions.size()));
    upload(d_unique, unique_ids.data(), unique_ids.size());
    upload(d_offsets, offsets.data(), offsets.size());
    upload(d_positions, positions.data(), positions.size());
    embedding_backward_kernel<<<grid_for(u_count * d, kBlock), kBlock>>>(
        dy.data(), d_unique.get(), d_offsets.get(), d_positions.get(), u_count,
        d, dtable_acc.data());
    check_launch();
}

}  // namespace titan::ops::cuda
