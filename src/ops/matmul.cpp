#include "ops/matmul.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

#include "ops/cpu/matmul.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

namespace {

// Checks ranks and shapes and returns the result shape (..., M, N).
Shape check_and_result_shape(const char* op, const Tensor& a, const Tensor& b,
                             bool ta, bool tb) {
    const std::string where =
        std::string("ops::") + op + ": " + to_string(a.shape()) + " x " +
        to_string(b.shape()) + " (transpose_a=" + (ta ? "true" : "false") +
        ", transpose_b=" + (tb ? "true" : "false") + ")";
    TITAN_CHECK(a.rank() >= 2 && b.rank() == a.rank(),
                where + ": operands must have equal rank between 2 and 4");
    for (int i = 0; i + 2 < a.rank(); ++i) {
        TITAN_CHECK(a.shape()[static_cast<std::size_t>(i)] ==
                        b.shape()[static_cast<std::size_t>(i)],
                    where + ": batch dimensions differ");
    }
    const std::int64_t m = ta ? a.dim(-1) : a.dim(-2);
    const std::int64_t ka = ta ? a.dim(-2) : a.dim(-1);
    const std::int64_t kb = tb ? b.dim(-1) : b.dim(-2);
    const std::int64_t n = tb ? b.dim(-2) : b.dim(-1);
    TITAN_CHECK(ka == kb, where + ": inner dimensions differ");
    Shape out(a.shape().begin(), a.shape().end() - 2);
    out.push_back(m);
    out.push_back(n);
    return out;
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a,
              bool transpose_b) {
    detail::require_cpu("matmul", {&a, &b});
    Tensor out = Tensor::zeros(
        check_and_result_shape("matmul", a, b, transpose_a, transpose_b),
        a.device());
    cpu::matmul_acc(out, a, b, transpose_a, transpose_b);
    return out;
}

void matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a,
                bool transpose_b) {
    detail::require_cpu("matmul_acc", {&out, &a, &b});
    const Shape expected =
        check_and_result_shape("matmul_acc", a, b, transpose_a, transpose_b);
    TITAN_CHECK(out.shape() == expected,
                "ops::matmul_acc: out has shape " + to_string(out.shape()) +
                    ", expected " + to_string(expected));
    cpu::matmul_acc(out, a, b, transpose_a, transpose_b);
}

}  // namespace titan::ops
