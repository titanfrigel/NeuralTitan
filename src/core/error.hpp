#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace titan {

// Base class of every exception the library throws.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Throws an Error whose message is "<file>:<line>: <message>".
[[noreturn]] void throw_error(const char* file, int line, std::string message);

// Throws an Error for a failed CUDA call; used by TITAN_CUDA_CHECK.
// Clears the sticky CUDA error state first so later calls start clean.
[[noreturn]] void throw_cuda_error(const char* file, int line, const char* expr,
                                   cudaError_t status);

}  // namespace titan

// Precondition check. `msg` is only evaluated when the check fails and may be
// a C string or a std::string.
#define TITAN_CHECK(cond, msg)                                             \
    do {                                                                   \
        if (!(cond)) {                                                     \
            ::titan::throw_error(                                          \
                __FILE__, __LINE__,                                        \
                std::string("check failed: " #cond ": ") + (msg));         \
        }                                                                  \
    } while (false)

// Wraps every CUDA runtime call and kernel launch check.
#define TITAN_CUDA_CHECK(expr)                                             \
    do {                                                                   \
        const cudaError_t titan_cuda_status = (expr);                      \
        if (titan_cuda_status != cudaSuccess) {                            \
            ::titan::throw_cuda_error(__FILE__, __LINE__, #expr,           \
                                      titan_cuda_status);                  \
        }                                                                  \
    } while (false)
