#pragma once

#include <cstdint>
#include <span>

// Comparisons of our outputs with the Hugging Face golden fixtures
// (tests/fixtures/{tiny_llama,smollm2}/golden.*). The rules are those of
// docs/math/hf-llama.md, "Tolerance".

namespace titan::test {

struct LogitCheck {
    double max_abs_diff;  // max |ours - ref|; infinity when a value is NaN
    double scale;         // max(1, max |ref|)
    bool ok;              // max_abs_diff <= tol * scale
};

// Compares two equally long sets of logits, in double. Every compared value
// of a prompt goes into one call, so the scale is that of the whole prompt.
// A NaN on either side fails. Throws titan::Error when the sizes differ.
LogitCheck compare_logits(std::span<const float> ours,
                          std::span<const float> ref, double tol);

struct GreedyCheck {
    // The number of leading steps where ours equals ref.
    std::int64_t matched;
    // Whether the first mismatch is at a step whose reference top-2 gap is
    // below tol (a near-tie, where FP32 rounding may pick either token).
    bool diverged_at_near_tie;
    // Every step matches, or the first mismatch is at a near-tie.
    bool ok;
};

// Compares greedy tokens step by step: ref has one token per reference step
// and ref_gaps the reference's top-2 logit gap at each of them. Comparison
// stops at the first step where ours differs from ref or has no token; ours's
// tokens beyond ref.size() are not compared. A run that ended on end-of-text
// has that token as its last step on both sides. Throws titan::Error when
// ref_gaps.size() != ref.size().
GreedyCheck compare_greedy(std::span<const std::int32_t> ours,
                           std::span<const std::int32_t> ref,
                           std::span<const double> ref_gaps, double tol);

}  // namespace titan::test
