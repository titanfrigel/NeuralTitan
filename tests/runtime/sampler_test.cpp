#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "harness/test.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "runtime/sampler.hpp"

using titan::Rng;
using titan::runtime::generate_naive;
using titan::runtime::sample;
using titan::runtime::sample_token;
using titan::runtime::SamplerConfig;
using titan::runtime::validate;

TITAN_TEST(Sampler, GreedyIsArgmax) {
    Rng rng(1);
    const std::vector<float> logits = {0.1f, 2.5f, -1.0f, 2.0f};
    TITAN_EXPECT_EQ(sample_token(logits, 0.0f, rng), 1);
    // Ties resolve to the lowest index.
    const std::vector<float> ties = {1.0f, 3.0f, 3.0f, 0.0f};
    TITAN_EXPECT_EQ(sample_token(ties, 0.0f, rng), 1);
    TITAN_EXPECT_THROWS(sample_token(std::vector<float>{}, 0.0f, rng),
                        titan::Error);
    TITAN_EXPECT_THROWS(sample_token(logits, -1.0f, rng), titan::Error);
}

TITAN_TEST(Sampler, TemperatureDistribution) {
    Rng rng(2);
    const std::vector<float> logits = {1.0f, 0.0f, -1.0f, 2.0f};
    double total = 0.0;
    std::vector<double> expected;
    for (float z : logits) {
        expected.push_back(std::exp(static_cast<double>(z)));
        total += expected.back();
    }
    const int draws = 20000;
    std::vector<int> counts(logits.size(), 0);
    for (int i = 0; i < draws; ++i) {
        ++counts[static_cast<std::size_t>(sample_token(logits, 1.0f, rng))];
    }
    for (std::size_t i = 0; i < logits.size(); ++i) {
        TITAN_EXPECT_NEAR(static_cast<double>(counts[i]) / draws,
                          expected[i] / total, 0.02);
    }
}

TITAN_TEST(Sampler, SeededIsDeterministic) {
    const std::vector<float> logits = {0.3f, 0.1f, 0.9f, 0.5f, 0.2f};
    Rng a(77);
    Rng b(77);
    for (int i = 0; i < 100; ++i) {
        TITAN_EXPECT_EQ(sample_token(logits, 0.8f, a),
                        sample_token(logits, 0.8f, b));
    }
}

TITAN_TEST(Sampler, GenerateNaiveExtendsPromptPastContext) {
    // Context 4, so after four tokens each step sees only the last four.
    titan::model::Model m(titan::model::llama_style(11, 8, 1, 2, 1, 4));
    Rng init(3);
    m.init(init);
    Rng a(5);
    Rng b(5);
    const std::vector<std::int32_t> prompt = {1, 2};
    const std::vector<std::int32_t> out = generate_naive(m, prompt, 8, 0.7f, a);
    TITAN_EXPECT_EQ(out.size(), std::size_t{10});
    TITAN_EXPECT_EQ(out[0], 1);
    TITAN_EXPECT_EQ(out[1], 2);
    for (std::int32_t id : out) {
        TITAN_EXPECT(id >= 0 && id < 11);
    }
    TITAN_EXPECT(generate_naive(m, prompt, 8, 0.7f, b) == out);
    TITAN_EXPECT_THROWS(generate_naive(m, {}, 1, 0.0f, a), titan::Error);
}

namespace {

// Draws `draws` tokens and returns how often each of n ids came up.
std::vector<double> frequencies(const std::vector<float>& logits,
                                const SamplerConfig& config, int draws,
                                std::uint64_t seed) {
    Rng rng(seed);
    std::vector<double> freq(logits.size(), 0.0);
    for (int i = 0; i < draws; ++i) {
        freq[static_cast<std::size_t>(sample(logits, config, rng))] += 1.0;
    }
    for (double& f : freq) {
        f /= draws;
    }
    return freq;
}

}  // namespace

TITAN_TEST(Sampler, GreedyIgnoresRng) {
    const std::vector<float> logits = {0.1f, 2.5f, -1.0f, 2.5f, 2.0f};
    Rng a(1);
    Rng b(2);
    const SamplerConfig greedy{0.0f, 0, 1.0f};
    TITAN_EXPECT_EQ(sample(logits, greedy, a), 1);
    TITAN_EXPECT_EQ(sample(logits, greedy, b), 1);
    // Neither generator advanced: both still match a fresh one.
    Rng fresh_a(1);
    Rng fresh_b(2);
    TITAN_EXPECT(a.next_u64() == fresh_a.next_u64());
    TITAN_EXPECT(b.next_u64() == fresh_b.next_u64());
    // Greedy ignores top_k and top_p as well.
    TITAN_EXPECT_EQ(sample(logits, SamplerConfig{0.0f, 2, 0.1f}, a), 1);
}

TITAN_TEST(Sampler, MatchesM1SampleToken) {
    // Outputs of the M1 sample_token (temperature 0.8) for these logits and
    // seeds, captured before the sampler was extended.
    const std::int32_t expected[200] = {
        15, 15, 2,  15, 2,  2,  15, 15, 3,  15, 15, 8,  6,  6,  15, 6,  15,
        15, 15, 3,  8,  4,  2,  2,  2,  2,  6,  6,  13, 9,  6,  2,  3,  15,
        6,  15, 13, 3,  2,  15, 2,  8,  8,  6,  2,  8,  6,  2,  11, 15, 2,
        8,  8,  11, 6,  2,  15, 2,  3,  2,  2,  15, 6,  6,  2,  2,  13, 2,
        15, 2,  14, 6,  8,  15, 3,  6,  1,  2,  3,  2,  14, 6,  11, 2,  6,
        15, 2,  6,  2,  6,  15, 14, 2,  2,  14, 15, 2,  15, 6,  2,  2,  1,
        15, 8,  6,  1,  15, 3,  6,  2,  15, 2,  10, 6,  15, 9,  8,  8,  2,
        6,  11, 6,  6,  2,  2,  6,  15, 15, 5,  6,  2,  6,  15, 15, 15, 14,
        11, 15, 6,  2,  15, 2,  11, 8,  13, 2,  8,  15, 11, 15, 6,  6,  2,
        2,  15, 6,  6,  6,  6,  15, 15, 15, 15, 15, 15, 6,  15, 15, 3,  6,
        15, 9,  2,  6,  2,  15, 8,  15, 6,  15, 15, 2,  2,  6,  6,  6,  15,
        3,  2,  15, 2,  6,  15, 8,  8,  2,  6,  15, 15, 1};
    Rng logit_rng(123);
    std::vector<float> logits(16);
    for (float& z : logits) {
        z = logit_rng.uniform() * 6.0f - 3.0f;
    }
    Rng old_rng(7);
    Rng new_rng(7);
    const SamplerConfig config{0.8f, 0, 1.0f};
    for (int i = 0; i < 200; ++i) {
        TITAN_EXPECT_EQ(sample_token(logits, 0.8f, old_rng), expected[i]);
        TITAN_EXPECT_EQ(sample(logits, config, new_rng), expected[i]);
    }
}

TITAN_TEST(Sampler, TopKNeverPicksOutside) {
    const std::vector<float> logits = {0.5f, 3.0f, -1.0f, 2.5f,  0.0f,
                                       1.0f, 2.0f, 0.1f,  -2.0f, 1.5f};
    const std::vector<double> freq =
        frequencies(logits, SamplerConfig{1.0f, 2, 1.0f}, 10000, 11);
    for (std::size_t i = 0; i < freq.size(); ++i) {
        if (i == 1 || i == 3) {
            TITAN_EXPECT(freq[i] > 0.0);
        } else {
            TITAN_EXPECT_EQ(freq[i], 0.0);
        }
    }
    // Ties at the cutoff keep the lower index.
    const std::vector<float> ties = {1.0f, 2.0f, 2.0f, 2.0f};
    const std::vector<double> tied =
        frequencies(ties, SamplerConfig{1.0f, 2, 1.0f}, 2000, 12);
    TITAN_EXPECT_EQ(tied[0], 0.0);
    TITAN_EXPECT_EQ(tied[3], 0.0);
    // top_k at or above the vocabulary size changes nothing.
    Rng a(5);
    Rng b(5);
    for (int i = 0; i < 50; ++i) {
        TITAN_EXPECT_EQ(sample(ties, SamplerConfig{1.0f, 99, 1.0f}, a),
                        sample(ties, SamplerConfig{1.0f, 0, 1.0f}, b));
    }
}

TITAN_TEST(Sampler, TopPKeepsMinimalSet) {
    // Probabilities {0.5, 0.3, 0.15, 0.05}; logits are their logs.
    const std::vector<float> logits = {std::log(0.5f), std::log(0.3f),
                                       std::log(0.15f), std::log(0.05f)};
    // 0.5 < 0.75 <= 0.5 + 0.3, so the minimal set is {0, 1}, renormalized to
    // {0.625, 0.375}.
    const std::vector<double> freq =
        frequencies(logits, SamplerConfig{1.0f, 0, 0.75f}, 20000, 21);
    TITAN_EXPECT_EQ(freq[2], 0.0);
    TITAN_EXPECT_EQ(freq[3], 0.0);
    TITAN_EXPECT_NEAR(freq[0], 0.625, 0.02);
    TITAN_EXPECT_NEAR(freq[1], 0.375, 0.02);
    // A tiny top_p keeps only the most likely token.
    const std::vector<double> top =
        frequencies(logits, SamplerConfig{1.0f, 0, 0.01f}, 1000, 22);
    TITAN_EXPECT_EQ(top[0], 1.0);
}

TITAN_TEST(Sampler, TopPAppliesAfterTopK) {
    // p = {0.5, 0.3, 0.15, 0.05}. top_k = 3 keeps {0, 1, 2} with mass 0.95,
    // renormalized to {0.526, 0.316, 0.158}. top_p = 0.83 then needs
    // 0.526 + 0.316 = 0.842 >= 0.83, so only {0, 1} survive. (Against the
    // full-vocabulary p the prefix sums are 0.5, 0.8, 0.95 and token 2 would
    // be kept.)
    const std::vector<float> logits = {std::log(0.5f), std::log(0.3f),
                                       std::log(0.15f), std::log(0.05f)};
    const std::vector<double> freq =
        frequencies(logits, SamplerConfig{1.0f, 3, 0.83f}, 20000, 41);
    TITAN_EXPECT_EQ(freq[2], 0.0);
    TITAN_EXPECT_EQ(freq[3], 0.0);
    TITAN_EXPECT_NEAR(freq[0], 0.625, 0.02);
    TITAN_EXPECT_NEAR(freq[1], 0.375, 0.02);
}

TITAN_TEST(Sampler, FrequenciesMatchSoftmax) {
    const std::vector<float> logits = {1.0f, 0.0f, -1.0f, 2.0f, 0.5f};
    double total = 0.0;
    for (float z : logits) {
        total += std::exp(static_cast<double>(z));
    }
    const std::vector<double> freq =
        frequencies(logits, SamplerConfig{1.0f, 0, 1.0f}, 20000, 31);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        TITAN_EXPECT_NEAR(
            freq[i], std::exp(static_cast<double>(logits[i])) / total, 0.015);
    }
}

TITAN_TEST(Sampler, RejectsBadConfig) {
    const std::vector<float> logits = {0.0f, 1.0f};
    Rng rng(1);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    TITAN_EXPECT_THROWS(validate(SamplerConfig{-0.1f, 0, 1.0f}), titan::Error);
    TITAN_EXPECT_THROWS(validate(SamplerConfig{nan, 0, 1.0f}), titan::Error);
    TITAN_EXPECT_THROWS(validate(SamplerConfig{inf, 0, 1.0f}), titan::Error);
    TITAN_EXPECT_THROWS(validate(SamplerConfig{1.0f, -1, 1.0f}), titan::Error);
    TITAN_EXPECT_THROWS(validate(SamplerConfig{1.0f, 0, 0.0f}), titan::Error);
    TITAN_EXPECT_THROWS(validate(SamplerConfig{1.0f, 0, 1.5f}), titan::Error);
    TITAN_EXPECT_THROWS(validate(SamplerConfig{1.0f, 0, nan}), titan::Error);
    validate(SamplerConfig{0.0f, 0, 1.0f});
    validate(SamplerConfig{1.0f, 5, 0.5f});
    TITAN_EXPECT_THROWS(sample(logits, SamplerConfig{1.0f, -2, 1.0f}, rng),
                        titan::Error);
    TITAN_EXPECT_THROWS(sample({}, SamplerConfig{}, rng), titan::Error);
    try {
        validate(SamplerConfig{1.0f, -1, 1.0f});
    } catch (const titan::Error& e) {
        TITAN_EXPECT(std::string(e.what()).find("top_k") != std::string::npos);
    }
}
