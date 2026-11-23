#pragma once

#include "core/tensor.hpp"

namespace titan::ops::cuda {

// CUDA matmul kernels on CUDA tensors. They do not validate; ops/matmul.hpp
// does. Like ops/cpu/matmul.hpp they take a of shape (..., A0, A1) and b of
// shape (..., B0, B1) with equal leading (batch) dimensions, where op(a) is
// (M, K) and op(b) is (K, N) after the optional transposes, and out is
// (..., M, N). All of them accumulate: out += op(a)·op(b). They launch on the
// default stream and return without waiting for the GPU.

// The GEMM implementation matmul_acc runs. It mirrors ops::GemmBackend
// (ops/matmul.hpp), which this layer may not include; ops/matmul.cpp maps one
// to the other.
enum class GemmBackend { cublas, naive, tiled };

// Runs the selected implementation. cublas calls cublasSgemmStridedBatched
// with beta = 1 on the row-major mapping C^T = op(b)^T · op(a)^T.
void matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a,
                bool transpose_b, GemmBackend backend);

// One thread per output element, a single float accumulator over k in
// increasing order.
void matmul_naive(Tensor& out, const Tensor& a, const Tensor& b,
                  bool transpose_a, bool transpose_b);

// 16 x 16 output tiles, one per 16 x 16 thread block, with the inputs staged
// through 16 x 16 shared-memory tiles. Same accumulation order as
// matmul_naive.
void matmul_tiled(Tensor& out, const Tensor& a, const Tensor& b,
                  bool transpose_a, bool transpose_b);

}  // namespace titan::ops::cuda
