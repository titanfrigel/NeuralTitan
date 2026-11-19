#include "core/error.hpp"

#include <utility>

namespace titan {

void throw_error(const char* file, int line, std::string message) {
    throw Error(std::string(file) + ":" + std::to_string(line) + ": " +
                message);
}

void throw_cuda_error(const char* file, int line, const char* expr,
                      cudaError_t status) {
    // Capture the strings before clearing: the sticky error would otherwise
    // poison every later CUDA call.
    const std::string name = cudaGetErrorName(status);
    const std::string description = cudaGetErrorString(status);
    cudaGetLastError();
    throw_error(file, line,
                std::string("CUDA call failed: ") + expr + ": " + name + ": " +
                    description);
}

}  // namespace titan
