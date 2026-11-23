#include "ops/layout.hpp"

#include <string>

#include "ops/cpu/layout.hpp"
#include "ops/cuda/layout.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

using detail::Backend;

namespace {

void require_rank4(const char* op, const Tensor& x) {
    TITAN_CHECK(x.rank() == 4, std::string("ops::") + op +
                                   ": input must have rank 4, got " +
                                   to_string(x.shape()));
}

void require_n_rep(const char* op, std::int64_t n_rep) {
    TITAN_CHECK(n_rep >= 1, std::string("ops::") + op +
                                ": n_rep must be >= 1, got " +
                                std::to_string(n_rep));
}

}  // namespace

Tensor swap_dims_1_2(const Tensor& x) {
    const Backend backend = detail::route("swap_dims_1_2", {&x});
    require_rank4("swap_dims_1_2", x);
    return backend == Backend::cuda ? cuda::swap_dims_1_2(x)
                                    : cpu::swap_dims_1_2(x);
}

Tensor repeat_kv(const Tensor& x, std::int64_t n_rep) {
    const Backend backend = detail::route("repeat_kv", {&x});
    require_rank4("repeat_kv", x);
    require_n_rep("repeat_kv", n_rep);
    return backend == Backend::cuda ? cuda::repeat_kv(x, n_rep)
                                    : cpu::repeat_kv(x, n_rep);
}

Tensor repeat_kv_backward(const Tensor& dy, std::int64_t n_rep) {
    const Backend backend = detail::route("repeat_kv_backward", {&dy});
    require_rank4("repeat_kv_backward", dy);
    require_n_rep("repeat_kv_backward", n_rep);
    TITAN_CHECK(dy.dim(1) % n_rep == 0,
                "ops::repeat_kv_backward: head count of " +
                    to_string(dy.shape()) + " is not divisible by n_rep " +
                    std::to_string(n_rep));
    return backend == Backend::cuda ? cuda::repeat_kv_backward(dy, n_rep)
                                    : cpu::repeat_kv_backward(dy, n_rep);
}

}  // namespace titan::ops
