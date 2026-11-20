#include "ops/cpu/elementwise.hpp"

#include <cmath>
#include <cstdint>

namespace titan::ops::cpu {

namespace {

// sqrt(2 / pi) and the cubic coefficient of the GELU tanh approximation.
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCubic = 0.044715f;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const float* pa = a.data();
    const float* pb = b.data();
    float* po = out.data();
    for (std::int64_t i = 0; i < a.numel(); ++i) po[i] = pa[i] + pb[i];
    return out;
}

void add_(Tensor& a, const Tensor& b) {
    float* pa = a.data();
    const float* pb = b.data();
    for (std::int64_t i = 0; i < a.numel(); ++i) pa[i] += pb[i];
}

void add_rowwise_(Tensor& y, const Tensor& b) {
    const std::int64_t cols = b.numel();
    const std::int64_t rows = y.numel() / cols;
    float* py = y.data();
    const float* pb = b.data();
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) py[r * cols + c] += pb[c];
    }
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const float* pa = a.data();
    const float* pb = b.data();
    float* po = out.data();
    for (std::int64_t i = 0; i < a.numel(); ++i) po[i] = pa[i] * pb[i];
    return out;
}

Tensor scale(const Tensor& x, float s) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const float* px = x.data();
    float* po = out.data();
    for (std::int64_t i = 0; i < x.numel(); ++i) po[i] = s * px[i];
    return out;
}

void scale_(Tensor& x, float s) {
    float* px = x.data();
    for (std::int64_t i = 0; i < x.numel(); ++i) px[i] *= s;
}

// y = x * sigmoid(x)
Tensor silu(const Tensor& x) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const float* px = x.data();
    float* po = out.data();
    for (std::int64_t i = 0; i < x.numel(); ++i) po[i] = px[i] * sigmoid(px[i]);
    return out;
}

// dy/dx = s * (1 + x * (1 - s)) with s = sigmoid(x); dx = dy * dy/dx.
Tensor silu_backward(const Tensor& x, const Tensor& dy) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const float* px = x.data();
    const float* pd = dy.data();
    float* po = out.data();
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        const float s = sigmoid(px[i]);
        po[i] = pd[i] * s * (1.0f + px[i] * (1.0f - s));
    }
    return out;
}

// y = 0.5 x (1 + tanh(u)), u = sqrt(2/pi) (x + 0.044715 x^3)
Tensor gelu(const Tensor& x) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const float* px = x.data();
    float* po = out.data();
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        const float v = px[i];
        const float u = kSqrt2OverPi * (v + kGeluCubic * v * v * v);
        po[i] = 0.5f * v * (1.0f + std::tanh(u));
    }
    return out;
}

// dy/dx = 0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3 * 0.044715 x^2),
// with t = tanh(u); dx = dy * dy/dx.
Tensor gelu_backward(const Tensor& x, const Tensor& dy) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const float* px = x.data();
    const float* pd = dy.data();
    float* po = out.data();
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        const float v = px[i];
        const float u = kSqrt2OverPi * (v + kGeluCubic * v * v * v);
        const float t = std::tanh(u);
        const float du = kSqrt2OverPi * (1.0f + 3.0f * kGeluCubic * v * v);
        po[i] = pd[i] * (0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du);
    }
    return out;
}

}  // namespace titan::ops::cpu
