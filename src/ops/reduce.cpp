#include "ops/reduce.hpp"

#include "ops/cpu/reduce.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

double sum(const Tensor& x) {
    detail::require_cpu("sum", {&x});
    return cpu::sum(x);
}

double sum_squares(const Tensor& x) {
    detail::require_cpu("sum_squares", {&x});
    return cpu::sum_squares(x);
}

Tensor row_max(const Tensor& x) {
    detail::require_cpu("row_max", {&x});
    return cpu::row_max(x);
}

Tensor sum_leading(const Tensor& x) {
    detail::require_cpu("sum_leading", {&x});
    return cpu::sum_leading(x);
}

bool all_finite(const Tensor& x) {
    detail::require_cpu("all_finite", {&x});
    return cpu::all_finite(x);
}

}  // namespace titan::ops
