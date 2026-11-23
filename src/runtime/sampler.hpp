#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/rng.hpp"
#include "model/model.hpp"

namespace titan::runtime {

// Sampling controls. temperature 0 means greedy; top_k 0 and top_p 1 disable
// those filters.
struct SamplerConfig {
    float temperature = 1.0f;
    std::int64_t top_k = 0;
    float top_p = 1.0f;
};

// Throws titan::Error naming the field for a temperature that is negative or
// non-finite, top_k < 0, or top_p outside (0, 1].
void validate(const SamplerConfig& config);

// Picks a token id from one position's logits. temperature == 0 returns the
// argmax (the lowest index among ties) and draws nothing from the rng.
// Otherwise: softmax(logits / temperature) in double, then top-k, then top-p,
// then an inverse-CDF draw with exactly one rng.uniform() over the kept tokens
// in index order (see docs/math/sampling.md). Throws titan::Error for empty
// logits or an invalid config.
std::int32_t sample(std::span<const float> logits, const SamplerConfig& config,
                    Rng& rng);

// sample(logits, {temperature, 0, 1.0f}, rng).
std::int32_t sample_token(std::span<const float> logits, float temperature,
                          Rng& rng);

// Extends `prompt` by n_new tokens and returns the prompt followed by the new
// tokens. Every step runs the whole of the last min(len, context_length)
// tokens through the model in inference mode (batch 1, no KV cache) and
// samples from the last position. Throws titan::Error for an empty prompt, a
// negative n_new or an invalid temperature.
std::vector<std::int32_t> generate_naive(model::Model& model,
                                         std::vector<std::int32_t> prompt,
                                         std::int64_t n_new, float temperature,
                                         Rng& rng);

}  // namespace titan::runtime
