#include "ops/rope.hpp"

#include <string>

#include "ops/cpu/rope.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

namespace {

void check_args(const char* op, const Tensor& x, std::int64_t pos_offset) {
    detail::require_cpu(op, {&x});
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
}

}  // namespace

Tensor rope(const Tensor& x, std::int64_t pos_offset, float theta) {
    check_args("rope", x, pos_offset);
    return cpu::rope(x, pos_offset, theta);
}

Tensor rope_backward(const Tensor& dy, std::int64_t pos_offset, float theta) {
    check_args("rope_backward", dy, pos_offset);
    return cpu::rope_backward(dy, pos_offset, theta);
}

}  // namespace titan::ops
