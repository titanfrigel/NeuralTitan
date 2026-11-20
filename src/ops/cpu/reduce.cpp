#include "ops/cpu/reduce.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace titan::ops::cpu {

double sum(const Tensor& x) {
    const float* p = x.data();
    double acc = 0.0;
    for (std::int64_t i = 0; i < x.numel(); ++i) acc += p[i];
    return acc;
}

double sum_squares(const Tensor& x) {
    const float* p = x.data();
    double acc = 0.0;
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        const double v = p[i];
        acc += v * v;
    }
    return acc;
}

Tensor row_max(const Tensor& x) {
    const std::int64_t cols = x.dim(-1);
    const std::int64_t rows = x.numel() / cols;
    Shape out_shape(x.shape().begin(), x.shape().end() - 1);
    if (out_shape.empty()) out_shape.push_back(1);
    Tensor out = Tensor::empty(out_shape, x.device());
    const float* p = x.data();
    float* po = out.data();
    for (std::int64_t r = 0; r < rows; ++r) {
        float m = p[r * cols];
        for (std::int64_t c = 1; c < cols; ++c) {
            if (p[r * cols + c] > m) m = p[r * cols + c];
        }
        po[r] = m;
    }
    return out;
}

Tensor sum_leading(const Tensor& x) {
    const std::int64_t cols = x.dim(-1);
    const std::int64_t rows = x.numel() / cols;
    // One double accumulator per column.
    std::vector<double> acc(static_cast<std::size_t>(cols), 0.0);
    const float* p = x.data();
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            acc[static_cast<std::size_t>(c)] += p[r * cols + c];
        }
    }
    Tensor out = Tensor::empty({cols}, x.device());
    float* po = out.data();
    for (std::int64_t c = 0; c < cols; ++c) {
        po[c] = static_cast<float>(acc[static_cast<std::size_t>(c)]);
    }
    return out;
}

bool all_finite(const Tensor& x) {
    const float* p = x.data();
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

}  // namespace titan::ops::cpu
