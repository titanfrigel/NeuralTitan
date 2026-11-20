#include <cstddef>
#include <cstdint>

#include "core/cuda_allocator.hpp"
#include "core/error.hpp"
#include "core/storage.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"

TITAN_TEST(Storage, CpuIsAligned) {
    titan::Storage storage(titan::Device::CPU, 100);
    TITAN_EXPECT(reinterpret_cast<std::uintptr_t>(storage.data()) % 64 == 0);
    TITAN_EXPECT_EQ(storage.bytes(), std::size_t{100});
    TITAN_EXPECT(storage.device() == titan::Device::CPU);
}

TITAN_TEST(Storage, ZeroBytesRejected) {
    TITAN_EXPECT_THROWS(titan::Storage(titan::Device::CPU, 0), titan::Error);
}

TITAN_TEST(Storage, CudaFreesToCache) {
    TITAN_SKIP_IF_NO_CUDA();
    auto& alloc = titan::CudaCachingAllocator::instance();
    alloc.release_cached();
    const std::size_t cached_before = alloc.cached_bytes();
    {
        titan::Storage storage(titan::Device::CUDA, 4096);
        TITAN_EXPECT(storage.data() != nullptr);
    }
    TITAN_EXPECT_EQ(alloc.cached_bytes() - cached_before, std::size_t{4096});
}
