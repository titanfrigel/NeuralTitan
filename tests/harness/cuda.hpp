#pragma once

#include "core/device.hpp"
#include "harness/test.hpp"

// Skips the running test when the machine has no CUDA device. Use it as a
// statement: TITAN_SKIP_IF_NO_CUDA();
#define TITAN_SKIP_IF_NO_CUDA()           \
    do {                                  \
        if (!::titan::cuda_available()) { \
            TITAN_SKIP("no CUDA device"); \
        }                                 \
    } while (false)
