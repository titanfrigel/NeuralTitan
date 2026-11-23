#include "ops/cross_entropy.hpp"

#include <string>

#include "ops/cpu/cross_entropy.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

float cross_entropy(const Tensor& logits, std::span<const std::int32_t> targets,
                    Tensor* dlogits_out) {
    detail::require_cpu("cross_entropy", {&logits});
    TITAN_CHECK(logits.rank() == 2,
                "ops::cross_entropy: logits must have rank 2 (N, V), got " +
                    to_string(logits.shape()));
    TITAN_CHECK(static_cast<std::int64_t>(targets.size()) == logits.dim(0),
                "ops::cross_entropy: got " + std::to_string(targets.size()) +
                    " targets for logits " + to_string(logits.shape()));
    const std::int64_t vocab = logits.dim(1);
    for (std::size_t i = 0; i < targets.size(); ++i) {
        TITAN_CHECK(targets[i] >= 0 && targets[i] < vocab,
                    "ops::cross_entropy: target " + std::to_string(targets[i]) +
                        " at index " + std::to_string(i) +
                        " is out of range [0, " + std::to_string(vocab) + ")");
    }
    return cpu::cross_entropy(logits, targets, dlogits_out);
}

}  // namespace titan::ops
