#include "ops/cpu/embedding.hpp"

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

namespace titan::ops::cpu {

Tensor embedding(const Tensor& table, std::span<const std::int32_t> ids,
                 Shape shape) {
    const std::int64_t d = table.dim(1);
    Tensor y = Tensor::empty(std::move(shape), table.device());
    const float* pt = table.data();
    float* py = y.data();
    for (std::size_t k = 0; k < ids.size(); ++k) {
        std::memcpy(py + static_cast<std::int64_t>(k) * d,
                    pt + static_cast<std::int64_t>(ids[k]) * d,
                    static_cast<std::size_t>(d) * sizeof(float));
    }
    return y;
}

void embedding_backward(Tensor& dtable_acc, std::span<const std::int32_t> ids,
                        const Tensor& dy) {
    const std::int64_t d = dtable_acc.dim(1);
    // Repeated ids sum several rows into one table row. The sums are collected
    // in a double scratch (in id order), then added to the accumulator once.
    std::vector<double> scratch(static_cast<std::size_t>(dtable_acc.numel()),
                                0.0);
    const float* pdy = dy.data();
    for (std::size_t k = 0; k < ids.size(); ++k) {
        double* row = scratch.data() + static_cast<std::int64_t>(ids[k]) * d;
        const float* src = pdy + static_cast<std::int64_t>(k) * d;
        for (std::int64_t j = 0; j < d; ++j) row[j] += src[j];
    }
    float* pg = dtable_acc.data();
    for (std::size_t i = 0; i < scratch.size(); ++i) {
        pg[i] += static_cast<float>(scratch[i]);
    }
}

}  // namespace titan::ops::cpu
