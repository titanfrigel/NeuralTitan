#include "runtime/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "core/error.hpp"
#include "nn/module.hpp"

namespace titan::runtime {

void validate(const SamplerConfig& config) {
    TITAN_CHECK(std::isfinite(config.temperature) && config.temperature >= 0.0f,
                "SamplerConfig: temperature must be finite and >= 0, got " +
                    std::to_string(config.temperature));
    TITAN_CHECK(config.top_k >= 0, "SamplerConfig: top_k must be >= 0, got " +
                                       std::to_string(config.top_k));
    TITAN_CHECK(config.top_p > 0.0f && config.top_p <= 1.0f,
                "SamplerConfig: top_p must be in (0, 1], got " +
                    std::to_string(config.top_p));
}

std::int32_t sample(std::span<const float> logits, const SamplerConfig& config,
                    Rng& rng) {
    TITAN_CHECK(!logits.empty(), "sample: empty logits");
    validate(config);
    const auto n = logits.size();
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    if (config.temperature == 0.0f) {
        return static_cast<std::int32_t>(best);
    }

    // w_i = exp((z_i - z_max) / T); p_i = w_i / sum_j w_j, in double. The
    // weights stay unnormalized: dropped tokens get weight 0 and the draw
    // scales u by the kept total, which is the renormalized inverse CDF.
    const double max_logit = logits[best];
    std::vector<double> w(n);
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = std::exp((static_cast<double>(logits[i]) - max_logit) /
                        static_cast<double>(config.temperature));
        total += w[i];
    }

    const bool use_top_k =
        config.top_k > 0 && static_cast<std::uint64_t>(config.top_k) < n;
    const bool use_top_p = config.top_p < 1.0f;
    if (use_top_k || use_top_p) {
        // Order by probability descending, lower index first on ties.
        std::vector<std::size_t> order(n);
        for (std::size_t i = 0; i < n; ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(),
                  [&w](std::size_t a, std::size_t b) {
                      return w[a] != w[b] ? w[a] > w[b] : a < b;
                  });
        std::size_t kept = n;
        if (use_top_k) {
            kept = static_cast<std::size_t>(config.top_k);
        }
        if (use_top_p) {
            // Shortest prefix of the kept tokens with probability mass >= p.
            double mass = 0.0;
            std::size_t prefix = 0;
            while (prefix < kept) {
                mass += w[order[prefix]] / total;
                ++prefix;
                if (mass >= static_cast<double>(config.top_p)) {
                    break;
                }
            }
            kept = prefix;
        }
        for (std::size_t r = kept; r < n; ++r) {
            w[order[r]] = 0.0;
        }
        total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            total += w[i];
        }
    }

    // Inverse CDF: the first kept index whose cumulative weight exceeds u.
    const double target = static_cast<double>(rng.uniform()) * total;
    double cumulative = 0.0;
    std::size_t last_kept = best;
    for (std::size_t i = 0; i < n; ++i) {
        if (w[i] == 0.0) {
            continue;
        }
        last_kept = i;
        cumulative += w[i];
        if (target < cumulative) {
            return static_cast<std::int32_t>(i);
        }
    }
    return static_cast<std::int32_t>(last_kept);
}

std::int32_t sample_token(std::span<const float> logits, float temperature,
                          Rng& rng) {
    return sample(logits, SamplerConfig{temperature, 0, 1.0f}, rng);
}

std::vector<std::int32_t> generate_naive(model::Model& model,
                                         std::vector<std::int32_t> prompt,
                                         std::int64_t n_new, float temperature,
                                         Rng& rng) {
    TITAN_CHECK(!prompt.empty(), "generate_naive: empty prompt");
    TITAN_CHECK(n_new >= 0, "generate_naive: n_new must be >= 0, got " +
                                std::to_string(n_new));
    const std::int64_t context = model.config().context_length;
    const std::int64_t vocab = model.config().vocab_size;
    std::vector<std::int32_t> tokens = std::move(prompt);
    for (std::int64_t i = 0; i < n_new; ++i) {
        const auto len = static_cast<std::int64_t>(tokens.size());
        const std::int64_t t = std::min(len, context);
        nn::ForwardContext ctx;
        ctx.training = false;
        const std::span<const std::int32_t> window(tokens.data() + (len - t),
                                                   static_cast<std::size_t>(t));
        const Tensor logits = model.forward(window, 1, t, ctx);
        const std::vector<float> all = logits.to_vector();
        const std::span<const float> last(all.data() + (t - 1) * vocab,
                                          static_cast<std::size_t>(vocab));
        tokens.push_back(sample_token(last, temperature, rng));
    }
    return tokens;
}

}  // namespace titan::runtime
