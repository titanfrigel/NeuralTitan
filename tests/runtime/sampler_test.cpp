#include <cmath>
#include <cstdint>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "harness/test.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "runtime/sampler.hpp"

using titan::Rng;
using titan::runtime::generate_naive;
using titan::runtime::sample_token;

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
