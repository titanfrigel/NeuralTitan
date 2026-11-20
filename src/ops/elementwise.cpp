#include "ops/elementwise.hpp"

#include "ops/cpu/elementwise.hpp"
#include "ops/cuda/elementwise.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

using detail::Backend;

Tensor add(const Tensor& a, const Tensor& b) {
    const Backend backend = detail::route("add", {&a, &b});
    detail::require_same_shape("add", a, b);
    return backend == Backend::cuda ? cuda::add(a, b) : cpu::add(a, b);
}

void add_(Tensor& a, const Tensor& b) {
    const Backend backend = detail::route("add_", {&a, &b});
    detail::require_same_shape("add_", a, b);
    if (backend == Backend::cuda) {
        cuda::add_(a, b);
    } else {
        cpu::add_(a, b);
    }
}

void add_rowwise_(Tensor& y, const Tensor& b) {
    const Backend backend = detail::route("add_rowwise_", {&y, &b});
    TITAN_CHECK(b.rank() == 1 && b.dim(0) == y.dim(-1),
                "ops::add_rowwise_: shape mismatch, y " + to_string(y.shape()) +
                    " vs b " + to_string(b.shape()));
    if (backend == Backend::cuda) {
        cuda::add_rowwise_(y, b);
    } else {
        cpu::add_rowwise_(y, b);
    }
}

Tensor mul(const Tensor& a, const Tensor& b) {
    const Backend backend = detail::route("mul", {&a, &b});
    detail::require_same_shape("mul", a, b);
    return backend == Backend::cuda ? cuda::mul(a, b) : cpu::mul(a, b);
}

Tensor scale(const Tensor& x, float s) {
    const Backend backend = detail::route("scale", {&x});
    return backend == Backend::cuda ? cuda::scale(x, s) : cpu::scale(x, s);
}

void scale_(Tensor& x, float s) {
    const Backend backend = detail::route("scale_", {&x});
    if (backend == Backend::cuda) {
        cuda::scale_(x, s);
    } else {
        cpu::scale_(x, s);
    }
}

Tensor silu(const Tensor& x) {
    const Backend backend = detail::route("silu", {&x});
    return backend == Backend::cuda ? cuda::silu(x) : cpu::silu(x);
}

Tensor silu_backward(const Tensor& x, const Tensor& dy) {
    const Backend backend = detail::route("silu_backward", {&x, &dy});
    detail::require_same_shape("silu_backward", x, dy);
    return backend == Backend::cuda ? cuda::silu_backward(x, dy)
                                    : cpu::silu_backward(x, dy);
}

Tensor gelu(const Tensor& x) {
    const Backend backend = detail::route("gelu", {&x});
    return backend == Backend::cuda ? cuda::gelu(x) : cpu::gelu(x);
}

Tensor gelu_backward(const Tensor& x, const Tensor& dy) {
    const Backend backend = detail::route("gelu_backward", {&x, &dy});
    detail::require_same_shape("gelu_backward", x, dy);
    return backend == Backend::cuda ? cuda::gelu_backward(x, dy)
                                    : cpu::gelu_backward(x, dy);
}

}  // namespace titan::ops
