#include "support/golden.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "core/error.hpp"

namespace titan::test {

LogitCheck compare_logits(std::span<const float> ours,
                          std::span<const float> ref, double tol) {
    TITAN_CHECK(ours.size() == ref.size(),
                "compare_logits: " + std::to_string(ours.size()) +
                    " values vs " + std::to_string(ref.size()) +
                    " reference values");
    double max_abs_diff = 0.0;
    double max_ref = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(ours[i]) - ref[i]);
        // std::max would drop a NaN; it must fail the check instead.
        if (std::isnan(diff)) {
            max_abs_diff = std::numeric_limits<double>::infinity();
        } else {
            max_abs_diff = std::max(max_abs_diff, diff);
        }
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i])));
    }
    // Logits are only defined up to their scale: a softmax over logits of
    // size ~20 cannot be expected to agree to an absolute 1e-3, but a scale
    // below 1 must not make the bound tighter than the tolerance itself.
    const double scale = std::max(1.0, max_ref);
    return {max_abs_diff, scale, max_abs_diff <= tol * scale};
}

GreedyCheck compare_greedy(std::span<const std::int32_t> ours,
                           std::span<const std::int32_t> ref,
                           std::span<const double> ref_gaps, double tol) {
    TITAN_CHECK(ref_gaps.size() == ref.size(),
                "compare_greedy: " + std::to_string(ref.size()) +
                    " reference tokens vs " + std::to_string(ref_gaps.size()) +
                    " gaps");
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (i >= ours.size() || ours[i] != ref[i]) {
            // At a near-tie the two best logits are closer than the logit
            // tolerance, so FP32 rounding alone may pick either token, and
            // everything after it continues a different text.
            const bool near_tie = ref_gaps[i] < tol;
            return {static_cast<std::int64_t>(i), near_tie, near_tie};
        }
    }
    return {static_cast<std::int64_t>(ref.size()), false, true};
}

}  // namespace titan::test
