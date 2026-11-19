#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace titan {

enum class Device { CPU, CUDA };

// "cpu" or "cuda".
std::string_view to_string(Device device);

// True when at least one CUDA device is usable. Cached after the first call;
// never throws.
bool cuda_available();

struct CudaDeviceInfo {
    std::string name;
    std::size_t total_memory_bytes;
    int compute_major;
    int compute_minor;
};

// Describes device 0, or std::nullopt when there is no GPU.
std::optional<CudaDeviceInfo> cuda_device_info();

}  // namespace titan
