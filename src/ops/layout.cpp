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

void kv_write(Tensor& cache, const Tensor& x, std::int64_t pos) {
    const Backend backend = detail::route("kv_write", {&cache, &x});
    require_rank4("kv_write", cache);
    require_rank4("kv_write", x);
    TITAN_CHECK(cache.dim(0) == x.dim(0) && cache.dim(1) == x.dim(1) &&
                    cache.dim(3) == x.dim(3),
                "ops::kv_write: batch, head and head-dim sizes of cache " +
                    to_string(cache.shape()) + " and x " +
                    to_string(x.shape()) + " differ");
    TITAN_CHECK(pos >= 0 && pos + x.dim(2) <= cache.dim(2),
                "ops::kv_write: rows " + std::to_string(pos) + ".." +
                    std::to_string(pos + x.dim(2)) + " do not fit the cache " +
                    to_string(cache.shape()));
    if (backend == Backend::cuda) {
        cuda::kv_write(cache, x, pos);
    } else {
        cpu::kv_write(cache, x, pos);
    }
}

Tensor kv_read(const Tensor& cache, std::int64_t len) {
    const Backend backend = detail::route("kv_read", {&cache});
    require_rank4("kv_read", cache);
    TITAN_CHECK(len >= 1 && len <= cache.dim(2),
                "ops::kv_read: len " + std::to_string(len) + " is outside 1.." +
                    std::to_string(cache.dim(2)) + " for the cache " +
                    to_string(cache.shape()));
    return backend == Backend::cuda ? cuda::kv_read(cache, len)
                                    : cpu::kv_read(cache, len);
}

}  // namespace titan::ops
