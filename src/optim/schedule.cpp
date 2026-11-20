#include "optim/schedule.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "core/error.hpp"

namespace titan::optim {

float lr_at(std::int64_t step, std::int64_t warmup, std::int64_t total,
            float max_lr, float min_lr) {
    TITAN_CHECK(warmup >= 1, "lr_at: warmup must be at least 1");
    TITAN_CHECK(total > warmup, "lr_at: total steps must exceed warmup");
    if (step < warmup) {
        return max_lr * static_cast<float>(step + 1) /
               static_cast<float>(warmup);
    }
    if (step >= total) return min_lr;
    // progress runs from 0 at step == warmup to 1 at step == total - 1.
    const double progress =
        static_cast<double>(step - warmup) /
        static_cast<double>(std::max<std::int64_t>(1, total - 1 - warmup));
    const double cosine = 0.5 * (1.0 + std::cos(std::numbers::pi * progress));
    return static_cast<float>(min_lr + (max_lr - min_lr) * cosine);
}

}  // namespace titan::optim
