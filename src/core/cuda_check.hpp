#pragma once

// CUDA error checking. Kept apart from core/error.hpp so that only the layers
// allowed to see the CUDA runtime (core and ops/cuda) pull in its header.

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace titan {

// Throws an Error for a failed CUDA call; used by TITAN_CUDA_CHECK.
// Clears the sticky CUDA error state first so later calls start clean.
[[noreturn]] void throw_cuda_error(const char* file, int line, const char* expr,
                                   cudaError_t status);

}  // namespace titan

// Wraps every CUDA runtime call and kernel launch check.
#define TITAN_CUDA_CHECK(expr)                                   \
    do {                                                         \
        const cudaError_t titan_cuda_status = (expr);            \
        if (titan_cuda_status != cudaSuccess) {                  \
            ::titan::throw_cuda_error(__FILE__, __LINE__, #expr, \
                                      titan_cuda_status);        \
        }                                                        \
    } while (false)
