#include <cstddef>
#include <string>

#include "core/cuda_allocator.hpp"
#include "core/error.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"

// The allocator is shared by every test in the process, so these tests only
// compare against state captured at their own start.

TITAN_TEST(Allocator, ReusesFreedBlock) {
    TITAN_SKIP_IF_NO_CUDA();
    auto& alloc = titan::CudaCachingAllocator::instance();
    alloc.release_cached();
    const std::size_t cached_before = alloc.cached_bytes();

    void* first = alloc.allocate(1000);
    alloc.deallocate(first, 1000);
    void* second = alloc.allocate(1000);

    TITAN_EXPECT(first == second);
    TITAN_EXPECT_EQ(alloc.cached_bytes(), cached_before);
    alloc.deallocate(second, 1000);
}

TITAN_TEST(Allocator, RoundsTo512) {
    TITAN_SKIP_IF_NO_CUDA();
    auto& alloc = titan::CudaCachingAllocator::instance();
    const std::size_t allocated_before = alloc.allocated_bytes();

    void* ptr = alloc.allocate(1);
    TITAN_EXPECT_EQ(alloc.allocated_bytes() - allocated_before, std::size_t{512});

    alloc.deallocate(ptr, 1);
    TITAN_EXPECT_EQ(alloc.allocated_bytes(), allocated_before);
}

TITAN_TEST(Allocator, OutOfMemoryThrowsAndRecovers) {
    TITAN_SKIP_IF_NO_CUDA();
    auto& alloc = titan::CudaCachingAllocator::instance();

    bool threw = false;
    try {
        alloc.allocate(std::size_t{1} << 40);
    } catch (const titan::Error& e) {
        threw = true;
        TITAN_EXPECT(std::string(e.what()).find("requested 1099511627776 bytes") !=
                     std::string::npos);
    }
    TITAN_EXPECT(threw);

    const std::size_t one_mb = std::size_t{1} << 20;
    void* ptr = alloc.allocate(one_mb);
    TITAN_EXPECT(ptr != nullptr);
    alloc.deallocate(ptr, one_mb);
}
