#include "runtime/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "core/error.hpp"
#include "nn/module.hpp"

namespace titan::runtime {

std::int32_t sample_token(std::span<const float> logits, float temperature,
                          Rng& rng) {
    TITAN_CHECK(!logits.empty(), "sample_token: empty logits");
    TITAN_CHECK(std::isfinite(temperature) && temperature >= 0.0f,
                "sample_token: temperature must be finite and >= 0, got " +
                    std::to_string(temperature));
    const auto n = static_cast<std::int64_t>(logits.size());
    std::int64_t best = 0;
    for (std::int64_t i = 1; i < n; ++i) {
        if (logits[static_cast<std::size_t>(i)] >
            logits[static_cast<std::size_t>(best)]) {
            best = i;
        }
    }
    if (temperature == 0.0f) {
        return static_cast<std::int32_t>(best);
    }

    // p_i = exp((z_i - z_max) / T) / sum_j exp((z_j - z_max) / T), in double.
    const double max_logit = logits[static_cast<std::size_t>(best)];
    std::vector<double> p(logits.size());
    double total = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp((static_cast<double>(logits[i]) - max_logit) /
                        static_cast<double>(temperature));
        total += p[i];
    }
    // Inverse CDF: the first index whose cumulative probability exceeds u.
    const double target = static_cast<double>(rng.uniform()) * total;
    double cumulative = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        cumulative += p[i];
        if (target < cumulative) {
            return static_cast<std::int32_t>(i);
        }
    }
    return static_cast<std::int32_t>(n - 1);
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
