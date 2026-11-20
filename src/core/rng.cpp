#include "core/rng.hpp"

#include <cmath>
#include <string>

#include "core/device.hpp"
#include "core/error.hpp"

namespace titan {

float Rng::uniform() {
    // 2^-24; (next_u64() >> 40) has 24 bits.
    return static_cast<float>(next_u64() >> 40) * (1.0f / 16777216.0f);
}

float Rng::normal() {
    if (has_cached_normal_) {
        has_cached_normal_ = false;
        return static_cast<float>(cached_normal_);
    }
    // u1 in (0, 1] so that log(u1) is finite; u2 in [0, 1). 53 bits each.
    constexpr double k2m53 = 1.0 / 9007199254740992.0;
    double u1 = 1.0 - static_cast<double>(next_u64() >> 11) * k2m53;
    double u2 = static_cast<double>(next_u64() >> 11) * k2m53;
    // z0 = r cos(theta), z1 = r sin(theta), r = sqrt(-2 ln u1), theta = 2 pi
    // u2.
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = kTwoPi * u2;
    cached_normal_ = r * std::sin(theta);
    has_cached_normal_ = true;
    return static_cast<float>(r * std::cos(theta));
}

std::int64_t Rng::randint(std::int64_t n) {
    TITAN_CHECK(n > 0, "randint needs n > 0, got " + std::to_string(n));
    const auto range = static_cast<std::uint64_t>(n);
    // Accept only draws below the largest multiple of `range` that fits in
    // 64 bits; each residue is then equally likely. 2^64 mod range equals
    // (UINT64_MAX % range + 1) % range.
    const std::uint64_t excess = (UINT64_MAX % range + 1) % range;
    const std::uint64_t limit = UINT64_MAX - excess;
    std::uint64_t x = next_u64();
    while (x > limit) {
        x = next_u64();
    }
    return static_cast<std::int64_t>(x % range);
}

void Rng::fill_normal(Tensor& t, float mean, float stddev) {
    TITAN_CHECK(t.device() == Device::CPU, "fill_normal needs a CPU tensor");
    float* p = t.data();
    const std::int64_t n = t.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        p[i] = mean + stddev * normal();
    }
}

}  // namespace titan
