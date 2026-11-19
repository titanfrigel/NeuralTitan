#include "core/cuda_allocator.hpp"

#include <string>

#include "core/error.hpp"

namespace titan {

namespace {
constexpr std::size_t kBlockGranularity = 512;
}  // namespace

CudaCachingAllocator& CudaCachingAllocator::instance() {
    // Leaked on purpose: a function-local static object would be destroyed
    // during process exit, when the CUDA context may already be torn down. The
    // driver reclaims all device memory at exit anyway.
    static CudaCachingAllocator* const allocator = new CudaCachingAllocator();
    return *allocator;
}

std::size_t CudaCachingAllocator::round_up(std::size_t bytes) {
    return (bytes + kBlockGranularity - 1) / kBlockGranularity *
           kBlockGranularity;
}

void* CudaCachingAllocator::allocate(std::size_t bytes) {
    TITAN_CHECK(bytes > 0, "cannot allocate zero bytes");
    const std::size_t size = round_up(bytes);

    std::lock_guard<std::mutex> lock(mutex_);

    // Fast path: reuse a cached block of exactly this size.
    const auto bucket = free_blocks_.find(size);
    if (bucket != free_blocks_.end() && !bucket->second.empty()) {
        void* ptr = bucket->second.back();
        bucket->second.pop_back();
        cached_bytes_ -= size;
        allocated_bytes_ += size;
        return ptr;
    }

    // Slow path: ask the driver.
    void* ptr = nullptr;
    cudaError_t status = cudaMalloc(&ptr, size);
    if (status == cudaErrorMemoryAllocation) {
        // The cache may be hoarding the memory we need. Clear the sticky error,
        // return all cached blocks to the driver and try once more.
        cudaGetLastError();
        release_cached_locked();
        status = cudaMalloc(&ptr, size);
        if (status == cudaErrorMemoryAllocation) {
            cudaGetLastError();
            throw_error(__FILE__, __LINE__,
                        "CUDA out of memory: requested " +
                            std::to_string(size) + " bytes (allocated " +
                            std::to_string(allocated_bytes_) + ", cached " +
                            std::to_string(cached_bytes_) + ")");
        }
    }
    TITAN_CUDA_CHECK(status);

    allocated_bytes_ += size;
    return ptr;
}

void CudaCachingAllocator::deallocate(void* ptr, std::size_t bytes) {
    if (ptr == nullptr) {
        return;
    }
    const std::size_t size = round_up(bytes);

    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_[size].push_back(ptr);
    cached_bytes_ += size;
    allocated_bytes_ -= size;
}

void CudaCachingAllocator::release_cached() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_cached_locked();
}

void CudaCachingAllocator::release_cached_locked() {
    for (auto& [size, blocks] : free_blocks_) {
        for (void* ptr : blocks) {
            TITAN_CUDA_CHECK(cudaFree(ptr));
        }
    }
    free_blocks_.clear();
    cached_bytes_ = 0;
}

std::size_t CudaCachingAllocator::cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cached_bytes_;
}

std::size_t CudaCachingAllocator::allocated_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_bytes_;
}

}  // namespace titan
