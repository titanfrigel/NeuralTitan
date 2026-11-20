#include "ops/elementwise.hpp"

#include "ops/cpu/elementwise.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

Tensor add(const Tensor& a, const Tensor& b) {
    detail::require_cpu("add", {&a, &b});
    detail::require_same_shape("add", a, b);
    return cpu::add(a, b);
}

void add_(Tensor& a, const Tensor& b) {
    detail::require_cpu("add_", {&a, &b});
    detail::require_same_shape("add_", a, b);
    cpu::add_(a, b);
}

void add_rowwise_(Tensor& y, const Tensor& b) {
    detail::require_cpu("add_rowwise_", {&y, &b});
    TITAN_CHECK(b.rank() == 1 && b.dim(0) == y.dim(-1),
                "ops::add_rowwise_: shape mismatch, y " + to_string(y.shape()) +
                    " vs b " + to_string(b.shape()));
    cpu::add_rowwise_(y, b);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    detail::require_cpu("mul", {&a, &b});
    detail::require_same_shape("mul", a, b);
    return cpu::mul(a, b);
}

Tensor scale(const Tensor& x, float s) {
    detail::require_cpu("scale", {&x});
    return cpu::scale(x, s);
}

void scale_(Tensor& x, float s) {
    detail::require_cpu("scale_", {&x});
    cpu::scale_(x, s);
}

Tensor silu(const Tensor& x) {
    detail::require_cpu("silu", {&x});
    return cpu::silu(x);
}

Tensor silu_backward(const Tensor& x, const Tensor& dy) {
    detail::require_cpu("silu_backward", {&x, &dy});
    detail::require_same_shape("silu_backward", x, dy);
    return cpu::silu_backward(x, dy);
}

Tensor gelu(const Tensor& x) {
    detail::require_cpu("gelu", {&x});
    return cpu::gelu(x);
}

Tensor gelu_backward(const Tensor& x, const Tensor& dy) {
    detail::require_cpu("gelu_backward", {&x, &dy});
    detail::require_same_shape("gelu_backward", x, dy);
    return cpu::gelu_backward(x, dy);
}

}  // namespace titan::ops
