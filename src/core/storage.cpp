#include "core/storage.hpp"

#include <new>

#include "core/cuda_allocator.hpp"
#include "core/error.hpp"

namespace titan {

namespace {
// 64 bytes is a cache line and the widest SIMD load the CPU kernels may use.
constexpr std::align_val_t kCpuAlignment{64};
}  // namespace

Storage::Storage(Device device, std::size_t bytes)
    : device_(device), bytes_(bytes), data_(nullptr) {
    TITAN_CHECK(bytes > 0, "storage size must be positive");
    if (device == Device::CUDA) {
        TITAN_CHECK(cuda_available(), "CUDA not available");
        data_ = CudaCachingAllocator::instance().allocate(bytes);
    } else {
        data_ = ::operator new(bytes, kCpuAlignment);
    }
}

Storage::~Storage() {
    if (device_ == Device::CUDA) {
        CudaCachingAllocator::instance().deallocate(data_, bytes_);
    } else {
        ::operator delete(data_, kCpuAlignment);
    }
}

}  // namespace titan
