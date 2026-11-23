#include "ops/softmax.hpp"

#include <string>

#include "ops/cpu/softmax.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

Tensor softmax(const Tensor& x) {
    detail::require_cpu("softmax", {&x});
    return cpu::softmax(x);
}

Tensor causal_softmax(const Tensor& scores, std::int64_t pos_offset) {
    detail::require_cpu("causal_softmax", {&scores});
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
    return cpu::causal_softmax(scores, pos_offset);
}

Tensor softmax_backward(const Tensor& y, const Tensor& dy) {
    detail::require_cpu("softmax_backward", {&y, &dy});
    detail::require_same_shape("softmax_backward", y, dy);
    return cpu::softmax_backward(y, dy);
}

}  // namespace titan::ops
