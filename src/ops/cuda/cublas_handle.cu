#include "ops/cuda/cublas_handle.cuh"

#include <string>

#include "core/error.hpp"

namespace titan::ops::cuda {

namespace {

cublasHandle_t create_handle() {
    cublasHandle_t handle = nullptr;
    TITAN_CUBLAS_CHECK(cublasCreate(&handle));
    TITAN_CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    return handle;
}

}  // namespace

cublasHandle_t cublas_handle() {
    // A function-local static is initialized once, thread-safely, on first
    // use. cublasHandle_t is a plain pointer, so nothing destroys it at exit.
    static const cublasHandle_t handle = create_handle();
    return handle;
}

void throw_cublas_error(const char* file, int line, const char* expr,
                        cublasStatus_t status) {
    throw_error(file, line,
                std::string("cuBLAS call failed: ") + expr + ": " +
                    cublasGetStatusName(status) + " (" +
                    std::to_string(static_cast<int>(status)) +
                    "): " + cublasGetStatusString(status));
}

}  // namespace titan::ops::cuda
