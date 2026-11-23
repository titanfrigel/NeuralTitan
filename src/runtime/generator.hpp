#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "core/rng.hpp"
#include "model/model.hpp"
#include "nn/kv_cache.hpp"
#include "runtime/sampler.hpp"

namespace titan::runtime {

// KV-cached generation, batch 1, in inference mode. generate() prefills the
// prompt into the cache, samples from the last position, then feeds each new
// token alone at pos_offset = cache length. When the cache is full it slides
// by half: it keeps the last max(1, context_length / 2) tokens, clears the
// cache and prefills them again at positions 0... See the "Generation loop"
// section of docs/math/kv-cache.md.
//
// The cache is built on the model's device at construction, so move the
// model to its device before constructing the Generator.
class Generator {
public:
    // Throws titan::Error for an invalid config (see validate()). The rng is
    // seeded once here and continues across generate() calls.
    Generator(model::Model& model, SamplerConfig config, std::uint64_t seed);

    // Returns the prompt followed by n_new new tokens and calls on_token (when
    // set) once per new token, in order, as soon as it is sampled. A prompt
    // longer than context_length is fed by its last context_length tokens,
    // but returned whole. Throws titan::Error for an empty prompt or a
    // negative n_new. Draws exactly one rng.uniform() per new token unless the
    // temperature is 0, in the same order as generate_naive, so before the
    // first slide both give the same tokens for the same seed.
    std::vector<std::int32_t> generate(
        std::vector<std::int32_t> prompt, std::int64_t n_new,
        const std::function<void(std::int32_t)>& on_token = {});

    // The cache, for inspection: its length is the number of positions the
    // next decode step attends over.
    const nn::KvCache& cache() const { return cache_; }

private:
    model::Model& model_;
    SamplerConfig config_;
    Rng rng_;
    nn::KvCache cache_;

    // Runs `ids` through the model at pos_offset = cache length, advances
    // the cache length and samples from the last position.
    std::int32_t feed(std::span<const std::int32_t> ids);
};

}  // namespace titan::runtime
