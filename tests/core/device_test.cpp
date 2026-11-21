#include "core/device.hpp"
#include "core/error.hpp"
#include "harness/test.hpp"

TITAN_TEST(Device, ToString) {
    TITAN_EXPECT_EQ(titan::to_string(titan::Device::CPU),
                    std::string_view("cpu"));
    TITAN_EXPECT_EQ(titan::to_string(titan::Device::CUDA),
                    std::string_view("cuda"));
}

TITAN_TEST(Device, InfoMatchesAvailability) {
    const auto info = titan::cuda_device_info();
    TITAN_EXPECT_EQ(info.has_value(), titan::cuda_available());
    if (info) {
        TITAN_EXPECT(info->compute_major >= 1);
        TITAN_EXPECT(info->total_memory_bytes > 0);
        // The bench derives peak compute and bandwidth from these.
        TITAN_EXPECT(info->sm_count >= 1);
        TITAN_EXPECT(info->clock_khz > 0);
        TITAN_EXPECT(info->memory_clock_khz > 0);
        TITAN_EXPECT(info->memory_bus_width_bits > 0);
    }
}

TITAN_TEST(Device, SynchronizeNeedsGpu) {
    if (titan::cuda_available()) {
        titan::cuda_synchronize();  // returns once the (idle) GPU is done
    } else {
        TITAN_EXPECT_THROWS(titan::cuda_synchronize(), titan::Error);
    }
}
