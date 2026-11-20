#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/rng.hpp"
#include "model/model.hpp"

namespace titan::runtime {

// Picks a token id from one position's logits. temperature == 0 returns the
// argmax (the lowest index among ties); otherwise draws from
// softmax(logits / temperature) by inverse CDF with one rng.uniform(). Throws
// titan::Error for empty logits or a negative or non-finite temperature.
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
