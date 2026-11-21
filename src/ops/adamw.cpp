#include "ops/adamw.hpp"

#include <string>

#include "ops/cpu/adamw.hpp"
#include "ops/cuda/adamw.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

void adamw_update(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                  const AdamWHyper& h) {
    const detail::Backend backend =
        detail::route("adamw_update", {&param, &grad, &m, &v});
    detail::require_same_shape("adamw_update", param, grad);
    detail::require_same_shape("adamw_update", param, m);
    detail::require_same_shape("adamw_update", param, v);
    TITAN_CHECK(h.step >= 1, "ops::adamw_update: step must be >= 1, got " +
                                 std::to_string(h.step));
    if (backend == detail::Backend::cuda) {
        cuda::adamw_update(param, grad, m, v, h.lr, h.beta1, h.beta2, h.eps,
                           h.weight_decay, h.step);
    } else {
        cpu::adamw_update(param, grad, m, v, h.lr, h.beta1, h.beta2, h.eps,
                          h.weight_decay, h.step);
    }
}

}  // namespace titan::ops
