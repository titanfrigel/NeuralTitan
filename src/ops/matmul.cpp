#include "ops/matmul.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

#include "ops/cpu/matmul.hpp"
#include "ops/cuda/matmul.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

namespace {

// Process-wide, like the CUDA caching allocator. Training is single-threaded,
// so it needs no synchronization.
GemmBackend g_gemm_backend = GemmBackend::cublas;

// ops/cuda may not include this header, so it declares its own mirror of
// GemmBackend; this is the one place that maps between the two.
cuda::GemmBackend to_cuda(GemmBackend backend) {
    switch (backend) {
        case GemmBackend::naive:
            return cuda::GemmBackend::naive;
        case GemmBackend::tiled:
            return cuda::GemmBackend::tiled;
        case GemmBackend::cublas:
            break;
    }
    return cuda::GemmBackend::cublas;
}

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

// out += op(a)·op(b) on the backend the shapes were validated for.
void run_matmul_acc(detail::Backend backend, Tensor& out, const Tensor& a,
                    const Tensor& b, bool ta, bool tb) {
    if (backend == detail::Backend::cuda) {
        cuda::matmul_acc(out, a, b, ta, tb, to_cuda(g_gemm_backend));
    } else {
        cpu::matmul_acc(out, a, b, ta, tb);
    }
}

}  // namespace

void set_gemm_backend(GemmBackend backend) { g_gemm_backend = backend; }

GemmBackend gemm_backend() { return g_gemm_backend; }

Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a,
              bool transpose_b) {
    const detail::Backend backend = detail::route("matmul", {&a, &b});
    // A zeroed out plus the accumulating kernel, on either device.
    Tensor out = Tensor::zeros(
        check_and_result_shape("matmul", a, b, transpose_a, transpose_b),
        a.device());
    run_matmul_acc(backend, out, a, b, transpose_a, transpose_b);
    return out;
}

void matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a,
                bool transpose_b) {
    const detail::Backend backend = detail::route("matmul_acc", {&out, &a, &b});
    const Shape expected =
        check_and_result_shape("matmul_acc", a, b, transpose_a, transpose_b);
    TITAN_CHECK(out.shape() == expected,
                "ops::matmul_acc: out has shape " + to_string(out.shape()) +
                    ", expected " + to_string(expected));
    run_matmul_acc(backend, out, a, b, transpose_a, transpose_b);
}

}  // namespace titan::ops
