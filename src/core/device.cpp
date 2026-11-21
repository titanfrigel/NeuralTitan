#include "core/device.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"
#include "core/error.hpp"

namespace titan {

std::string_view to_string(Device device) {
    switch (device) {
        case Device::CPU:
            return "cpu";
        case Device::CUDA:
            return "cuda";
    }
    return "unknown";
}

bool cuda_available() {
    static const bool available = [] {
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess) {
            cudaGetLastError();  // clear the sticky error
            return false;
        }
        return count >= 1;
    }();
    return available;
}

std::optional<CudaDeviceInfo> cuda_device_info() {
    if (!cuda_available()) {
        return std::nullopt;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        cudaGetLastError();
        return std::nullopt;
    }
    return CudaDeviceInfo{prop.name,
                          prop.totalGlobalMem,
                          prop.major,
                          prop.minor,
                          prop.multiProcessorCount,
                          prop.clockRate,
                          prop.memoryClockRate,
                          prop.memoryBusWidth};
}

void cuda_synchronize() {
    TITAN_CHECK(cuda_available(), "no CUDA device available");
    TITAN_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace titan
