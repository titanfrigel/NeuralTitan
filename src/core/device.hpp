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
    int sm_count;               // streaming multiprocessors
    int clock_khz;              // peak SM clock
    int memory_clock_khz;       // peak memory clock
    int memory_bus_width_bits;  // global memory bus width
};

// Describes device 0, or std::nullopt when there is no GPU.
std::optional<CudaDeviceInfo> cuda_device_info();

// Blocks the host until all work queued on the GPU has finished, so that a
// host timer around it measures GPU work. Throws titan::Error ("no CUDA
// device available") without a GPU, or when earlier GPU work failed.
void cuda_synchronize();

}  // namespace titan
