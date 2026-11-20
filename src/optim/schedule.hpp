#pragma once

#include <cstdint>

namespace titan::optim {

// Learning rate at `step` (steps start at 0): linear warmup from max_lr/warmup
// to max_lr at step warmup - 1, then cosine decay to min_lr at step total - 1,
// and min_lr afterwards. Requires warmup >= 1 and total > warmup.
float lr_at(std::int64_t step, std::int64_t warmup, std::int64_t total,
            float max_lr, float min_lr);

}  // namespace titan::optim
