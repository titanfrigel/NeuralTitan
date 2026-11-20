#include "nn/init.hpp"

#include <algorithm>

#include "core/error.hpp"

namespace titan::nn {

void init_normal(Parameter& p, Rng& rng, float stddev) {
    rng.fill_normal(p.value, 0.0f, stddev);
}

void init_constant(Parameter& p, float value) {
    TITAN_CHECK(p.value.device() == Device::CPU,
                "init_constant: parameter " + p.name + " is not on the CPU");
    float* data = p.value.data();
    std::fill(data, data + p.value.numel(), value);
}

}  // namespace titan::nn
