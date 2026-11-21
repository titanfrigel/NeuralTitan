#include "ops/rope.hpp"

#include <string>

#include "ops/cpu/rope.hpp"
#include "ops/cuda/rope.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

using detail::Backend;

namespace {

Backend check_args(const char* op, const Tensor& x, std::int64_t pos_offset) {
    const Backend backend = detail::route(op, {&x});
    TITAN_CHECK(x.rank() == 4,
                std::string("ops::") + op +
                    ": input must have rank 4 (B, H, T, hd), got " +
                    to_string(x.shape()));
    TITAN_CHECK(x.dim(-1) % 2 == 0,
                std::string("ops::") + op +
                    ": head dimension must be even, got shape " +
                    to_string(x.shape()));
    TITAN_CHECK(pos_offset >= 0, std::string("ops::") + op +
                                     ": pos_offset must be >= 0, got " +
                                     std::to_string(pos_offset));
    return backend;
}

}  // namespace

Tensor rope(const Tensor& x, std::int64_t pos_offset, float theta) {
    const Backend backend = check_args("rope", x, pos_offset);
    return backend == Backend::cuda ? cuda::rope(x, pos_offset, theta)
                                    : cpu::rope(x, pos_offset, theta);
}

Tensor rope_backward(const Tensor& dy, std::int64_t pos_offset, float theta) {
    const Backend backend = check_args("rope_backward", dy, pos_offset);
    return backend == Backend::cuda ? cuda::rope_backward(dy, pos_offset, theta)
                                    : cpu::rope_backward(dy, pos_offset, theta);
}

}  // namespace titan::ops
