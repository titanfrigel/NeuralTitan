#include "ops/reduce.hpp"

#include "ops/cpu/reduce.hpp"
#include "ops/cuda/reduce.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

double sum(const Tensor& x) {
    const detail::Backend backend = detail::route("sum", {&x});
    return backend == detail::Backend::cuda ? cuda::sum(x) : cpu::sum(x);
}

double sum_squares(const Tensor& x) {
    const detail::Backend backend = detail::route("sum_squares", {&x});
    return backend == detail::Backend::cuda ? cuda::sum_squares(x)
                                            : cpu::sum_squares(x);
}

Tensor row_max(const Tensor& x) {
    const detail::Backend backend = detail::route("row_max", {&x});
    return backend == detail::Backend::cuda ? cuda::row_max(x)
                                            : cpu::row_max(x);
}

Tensor sum_leading(const Tensor& x) {
    const detail::Backend backend = detail::route("sum_leading", {&x});
    return backend == detail::Backend::cuda ? cuda::sum_leading(x)
                                            : cpu::sum_leading(x);
}

bool all_finite(const Tensor& x) {
    const detail::Backend backend = detail::route("all_finite", {&x});
    return backend == detail::Backend::cuda ? cuda::all_finite(x)
                                            : cpu::all_finite(x);
}

}  // namespace titan::ops
