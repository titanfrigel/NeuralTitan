#include "ops/cpu/softmax.hpp"

#include <cmath>

namespace titan::ops::cpu {

namespace {

// Softmax of the first `valid` entries of a row of length `cols`; the rest are
// written as exactly 0. The maximum and the sum run over the valid entries
// only, so a masked entry never influences the result.
void softmax_row(const float* x, float* y, std::int64_t cols,
                 std::int64_t valid) {
    float m = x[0];
    for (std::int64_t j = 1; j < valid; ++j) {
        if (x[j] > m) m = x[j];
    }
    double total = 0.0;
    for (std::int64_t j = 0; j < valid; ++j) {
        const float e = std::exp(x[j] - m);
        y[j] = e;
        total += e;
    }
    const float inv = static_cast<float>(1.0 / total);
    for (std::int64_t j = 0; j < valid; ++j) y[j] *= inv;
    for (std::int64_t j = valid; j < cols; ++j) y[j] = 0.0f;
}

}  // namespace

Tensor softmax(const Tensor& x) {
    const std::int64_t cols = x.dim(-1);
    const std::int64_t rows = x.numel() / cols;
    Tensor out = Tensor::empty(x.shape(), x.device());
    for (std::int64_t r = 0; r < rows; ++r) {
        softmax_row(x.data() + r * cols, out.data() + r * cols, cols, cols);
    }
    return out;
}

Tensor causal_softmax(const Tensor& scores, std::int64_t pos_offset) {
    const std::int64_t tk = scores.dim(-1);
    const std::int64_t tq = scores.dim(-2);
    const std::int64_t rows = scores.numel() / tk;
    Tensor out = Tensor::empty(scores.shape(), scores.device());
    for (std::int64_t r = 0; r < rows; ++r) {
        const std::int64_t i = r % tq;
        // Keys 0..i+pos_offset are visible; Tk >= Tq + pos_offset keeps this
        // within the row.
        const std::int64_t valid = i + pos_offset + 1;
        softmax_row(scores.data() + r * tk, out.data() + r * tk, tk, valid);
    }
    return out;
}

Tensor softmax_backward(const Tensor& y, const Tensor& dy) {
    const std::int64_t cols = y.dim(-1);
    const std::int64_t rows = y.numel() / cols;
    Tensor dx = Tensor::empty(y.shape(), y.device());
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* py = y.data() + r * cols;
        const float* pd = dy.data() + r * cols;
        float* px = dx.data() + r * cols;
        double dot = 0.0;
        for (std::int64_t j = 0; j < cols; ++j) {
            dot += static_cast<double>(pd[j]) * py[j];
        }
        for (std::int64_t j = 0; j < cols; ++j) {
            px[j] = static_cast<float>(py[j] * (pd[j] - dot));
        }
    }
    return dx;
}

}  // namespace titan::ops::cpu
