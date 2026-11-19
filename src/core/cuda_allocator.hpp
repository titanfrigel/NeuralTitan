#pragma once

#include <cstddef>
#include <map>
#include <mutex>
#include <vector>

namespace titan {

// Process-wide caching allocator for CUDA device memory.
//
// cudaMalloc and cudaFree are slow and cudaFree synchronizes the whole device.
// A training loop allocates and frees tensors of the same few sizes every step,
// so calling them each time would dominate the runtime. Instead, freed blocks
// are kept in a free list keyed by their size and handed out again on the next
// allocation of that size. Memory only goes back to the driver when
// release_cached() is called, or when a cudaMalloc fails and a retry may
// succeed once the cache is emptied.
//
// Sizes are rounded up to a multiple of 512 bytes, so near-equal requests share
// a bucket and every block stays aligned for vectorized access. Reuse is by
// exact rounded size only; there is no splitting or coalescing, which keeps the
// bookkeeping easy to follow.
class CudaCachingAllocator {
public:
    static CudaCachingAllocator& instance();

    CudaCachingAllocator(const CudaCachingAllocator&) = delete;
    CudaCachingAllocator& operator=(const CudaCachingAllocator&) = delete;

    // Returns a device block of at least `bytes` bytes. Throws titan::Error
    // ("CUDA out of memory: ...") when the device cannot provide it even after
    // the cache has been released.
    void* allocate(std::size_t bytes);

    // Returns a block to the free list. `bytes` must be the size that was
    // passed to allocate; it selects the bucket without a pointer lookup.
    void deallocate(void* ptr, std::size_t bytes);

    // Gives every cached (free) block back to the driver.
    void release_cached();

    // Bytes sitting in the free lists, ready for reuse.
    std::size_t cached_bytes() const;

    // Bytes currently handed out to callers (rounded sizes).
    std::size_t allocated_bytes() const;

private:
    CudaCachingAllocator() = default;
    // Never runs: the instance is intentionally leaked (see instance()).
    ~CudaCachingAllocator() = default;

    static std::size_t round_up(std::size_t bytes);
    void release_cached_locked();

    mutable std::mutex mutex_;
    std::map<std::size_t, std::vector<void*>>
        free_blocks_;  // rounded size -> blocks
    std::size_t cached_bytes_ = 0;
    std::size_t allocated_bytes_ = 0;
};

}  // namespace titan
