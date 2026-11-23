#pragma once

#include "core/tensor.hpp"

namespace titan::ops {

// The GEMM implementation CUDA matmuls run (ADR 0007): cuBLAS, the default,
// or the project's own naive and tiled kernels. CPU matmuls ignore it.
enum class GemmBackend { cublas, naive, tiled };

// Selects the GEMM backend for every later CUDA matmul in the process. The
// choice is process-wide state, not synchronized: set it before training
// starts, from one thread.
void set_gemm_backend(GemmBackend backend);
GemmBackend gemm_backend();

// Matrix product of tensors of rank 2 to 4, all on one device. The leading
// dimensions are batch dimensions and must be equal in a and b. With the
// optional transposes applied to the last two dimensions, op(a) is
// (..., M, K) and op(b) is (..., K, N); the result is (..., M, N).
Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a = false,
              bool transpose_b = false);

// out += op(a)·op(b). out must already have the result shape. Used to
// accumulate weight gradients.
void matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a,
                bool transpose_b);

}  // namespace titan::ops
