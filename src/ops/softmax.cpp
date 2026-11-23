#include "ops/softmax.hpp"

#include <string>

#include "ops/cpu/softmax.hpp"
#include "ops/cuda/softmax.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

using detail::Backend;

Tensor softmax(const Tensor& x) {
    const Backend backend = detail::route("softmax", {&x});
    return backend == Backend::cuda ? cuda::softmax(x) : cpu::softmax(x);
}

Tensor causal_softmax(const Tensor& scores, std::int64_t pos_offset) {
    const Backend backend = detail::route("causal_softmax", {&scores});
    TITAN_CHECK(scores.rank() >= 2,
                "ops::causal_softmax: scores must have rank >= 2, got " +
                    to_string(scores.shape()));
    TITAN_CHECK(pos_offset >= 0,
                "ops::causal_softmax: pos_offset must be >= 0, got " +
                    std::to_string(pos_offset));
    const std::int64_t tq = scores.dim(-2);
    const std::int64_t tk = scores.dim(-1);
    TITAN_CHECK(tk >= tq + pos_offset,
                "ops::causal_softmax: need Tk >= Tq + pos_offset, got shape " +
                    to_string(scores.shape()) + " with pos_offset " +
                    std::to_string(pos_offset));
    return backend == Backend::cuda ? cuda::causal_softmax(scores, pos_offset)
                                    : cpu::causal_softmax(scores, pos_offset);
}

Tensor softmax_backward(const Tensor& y, const Tensor& dy) {
    const Backend backend = detail::route("softmax_backward", {&y, &dy});
    detail::require_same_shape("softmax_backward", y, dy);
    return backend == Backend::cuda ? cuda::softmax_backward(y, dy)
                                    : cpu::softmax_backward(y, dy);
}

}  // namespace titan::ops
