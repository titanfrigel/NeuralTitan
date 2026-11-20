#include "core/device.hpp"
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
    }
}
