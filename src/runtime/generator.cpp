#include "runtime/generator.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "nn/module.hpp"

namespace titan::runtime {

namespace {

SamplerConfig validated(const SamplerConfig& config) {
    validate(config);
    return config;
}

}  // namespace

Generator::Generator(model::Model& model, SamplerConfig config,
                     std::uint64_t seed)
    : model_(model),
      config_(validated(config)),
      rng_(seed),
      cache_(model.make_kv_cache(1)) {}

std::vector<std::int32_t> Generator::generate(
    std::vector<std::int32_t> prompt, std::int64_t n_new,
    const std::function<void(std::int32_t)>& on_token,
    std::optional<std::int32_t> stop_id) {
    TITAN_CHECK(!prompt.empty(), "Generator::generate: empty prompt");
    TITAN_CHECK(n_new >= 0, "Generator::generate: n_new must be >= 0, got " +
                                std::to_string(n_new));
    std::vector<std::int32_t> tokens = std::move(prompt);
    if (n_new == 0) {
        return tokens;
    }
    const std::int64_t context = model_.config().context_length;
    // After a slide the cache holds this many tokens, so a full cache leaves
    // room for at least one more decode step.
    const std::int64_t keep = std::max<std::int64_t>(1, context / 2);

    // The last `n` tokens of the sequence so far.
    const auto tail = [&tokens](std::int64_t n) {
        return std::span<const std::int32_t>(tokens).last(
            static_cast<std::size_t>(n));
    };

    // Prefill: the last min(len, context) prompt tokens at positions 0...
    cache_.clear();
    std::int32_t next =
        feed(tail(std::min(static_cast<std::int64_t>(tokens.size()), context)));
    for (std::int64_t i = 0;; ++i) {
        if (next == stop_id) {
            break;
        }
        tokens.push_back(next);
        if (on_token) {
            on_token(next);
        }
        if (i + 1 == n_new) {
            break;
        }
        if (cache_.length() == context) {
            // Slide by half: re-prefill the last `keep` tokens, which end
            // with `next`, at positions 0...
            cache_.clear();
            next = feed(tail(keep));
        } else {
            // Decode: `next` alone at position cache length.
            next = feed(tail(1));
        }
    }
    return tokens;
}

std::int32_t Generator::feed(std::span<const std::int32_t> ids) {
    const auto t = static_cast<std::int64_t>(ids.size());
    nn::ForwardContext ctx;
    ctx.training = false;
    ctx.pos_offset = cache_.length();
    ctx.kv_cache = &cache_;
    const Tensor logits = model_.forward(ids, 1, t, ctx);
    cache_.set_length(ctx.pos_offset + t);
    // Sampling runs on host floats: copy the logits (1, t, vocab) and take
    // the last row.
    const std::int64_t vocab = model_.config().vocab_size;
    const std::vector<float> all = logits.to_vector();
    return sample(std::span<const float>(all).subspan(
                      static_cast<std::size_t>((t - 1) * vocab),
                      static_cast<std::size_t>(vocab)),
                  config_, rng_);
}

}  // namespace titan::runtime
