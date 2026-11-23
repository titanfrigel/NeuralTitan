#pragma once

// The process-wide cuBLAS handle and the cuBLAS error check. CUDA-only:
// included by .cu files of ops/cuda, never by the plain C++ headers of this
// directory, so no other layer sees a cuBLAS type.

#include <cublas_v2.h>

namespace titan::ops::cuda {

// The handle every cuBLAS call uses. Created on first use with
// CUBLAS_PEDANTIC_MATH, so FP32 GEMMs never use TF32 tensor cores and stay
// comparable with the CPU reference. Like the caching allocator it is never
// destroyed: the driver may already be shutting down when static destructors
// run.
cublasHandle_t cublas_handle();

// Throws an Error for a failed cuBLAS call; used by TITAN_CUBLAS_CHECK. The
// message names the call, the status name and number, and file:line.
[[noreturn]] void throw_cublas_error(const char* file, int line,
                                     const char* expr, cublasStatus_t status);

}  // namespace titan::ops::cuda

// Wraps every cuBLAS call.
#define TITAN_CUBLAS_CHECK(expr)                                              \
    do {                                                                      \
        const cublasStatus_t titan_cublas_status = (expr);                    \
        if (titan_cublas_status != CUBLAS_STATUS_SUCCESS) {                   \
            ::titan::ops::cuda::throw_cublas_error(__FILE__, __LINE__, #expr, \
                                                   titan_cublas_status);      \
        }                                                                     \
    } while (false)
