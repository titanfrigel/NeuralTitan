#pragma once

#include "core/device.hpp"
#include "harness/test.hpp"

// Skips the running test when the machine has no CUDA device.
#define TITAN_SKIP_IF_NO_CUDA()                  \
    if (!::titan::cuda_available()) {            \
        TITAN_SKIP("no CUDA device");            \
    }
