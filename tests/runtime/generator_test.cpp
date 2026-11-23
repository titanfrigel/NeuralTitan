// The KV-cached Generator against the uncached generate_naive: the same tokens
// for the same seed until the first slide, streaming, the slide past the
// context, the long-prompt tail and the input checks.

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/rng.hpp"
#include "harness/test.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "nn/kv_cache.hpp"
#include "runtime/generator.hpp"
#include "runtime/sampler.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Rng;
using titan::model::llama_style;
using titan::model::Model;
using titan::runtime::generate_naive;
using titan::runtime::Generator;
using titan::runtime::SamplerConfig;

// A copy would share the tensors but keep its own length, so two copies would
// silently overwrite each other's rows.
static_assert(!std::is_copy_constructible_v<titan::nn::KvCache>);
static_assert(!std::is_copy_assignable_v<titan::nn::KvCache>);
static_assert(std::is_move_constructible_v<titan::nn::KvCache>);

namespace {

constexpr std::int64_t kVocab = 65;

void init_model(Model& m, std::uint64_t seed) {
    Rng rng(seed);
    m.init(rng);
}

// The M1 model-test shape (dim 16, 2 layers, 2 heads over 1 KV head) with the
// given context length.
titan::model::ModelConfig config_with_context(std::int64_t context) {
    return llama_style(kVocab, 16, 2, 2, 1, context);
}

const std::vector<std::int32_t> kPrompt = {3, 14, 15, 9, 26};

void expect_greedy_matches_naive(Model& m) {
    Rng unused(0);  // greedy draws nothing
    const std::vector<std::int32_t> want =
        generate_naive(m, kPrompt, 40, 0.0f, unused);
    Generator gen(m, SamplerConfig{0.0f, 0, 1.0f}, 0);
    const std::vector<std::int32_t> got = gen.generate(kPrompt, 40);
    TITAN_EXPECT_EQ(got.size(), std::size_t{45});
    TITAN_EXPECT(got == want);
}

}  // namespace

TITAN_TEST(Generator, GreedyMatchesNaive) {
    Model m(config_with_context(64));
    init_model(m, 70);
    expect_greedy_matches_naive(m);
}

TITAN_TEST(Generator, GreedyMatchesNaiveOnCuda) {
    TITAN_REQUIRE_CUDA();
    Model m(config_with_context(64));
    init_model(m, 70);
    m.to(Device::CUDA);
    expect_greedy_matches_naive(m);
}

TITAN_TEST(Generator, SampledMatchesNaiveOnCpu) {
    Model m(config_with_context(64));
    init_model(m, 71);
    Rng rng(123);
    const std::vector<std::int32_t> want =
        generate_naive(m, kPrompt, 40, 0.8f, rng);
    Generator gen(m, SamplerConfig{0.8f, 0, 1.0f}, 123);
    TITAN_EXPECT(gen.generate(kPrompt, 40) == want);
}

TITAN_TEST(Generator, StreamsEveryToken) {
    Model m(config_with_context(64));
    init_model(m, 72);
    Generator gen(m, SamplerConfig{0.9f, 10, 0.95f}, 5);
    std::vector<std::int32_t> streamed;
    const std::vector<std::int32_t> out = gen.generate(
        kPrompt, 20, [&](std::int32_t id) { streamed.push_back(id); });
    TITAN_EXPECT_EQ(out.size(), std::size_t{25});
    TITAN_EXPECT(std::vector<std::int32_t>(out.begin() + 5, out.end()) ==
                 streamed);
    TITAN_EXPECT(std::vector<std::int32_t>(out.begin(), out.begin() + 5) ==
                 kPrompt);
    // No new tokens: the prompt back and no callback.
    streamed.clear();
    TITAN_EXPECT(gen.generate(kPrompt, 0, [&](std::int32_t id) {
        streamed.push_back(id);
    }) == kPrompt);
    TITAN_EXPECT(streamed.empty());
}

TITAN_TEST(Generator, SlidesPastContext) {
    constexpr std::int64_t kContext = 16;
    constexpr std::int64_t kNew = 50;
    Model m(config_with_context(kContext));
    init_model(m, 73);
    Rng rng(9);
    const std::vector<std::int32_t> naive =
        generate_naive(m, kPrompt, kNew, 0.8f, rng);

    Generator gen(m, SamplerConfig{0.8f, 0, 1.0f}, 9);
    std::int64_t calls = 0;
    std::int64_t max_length = 0;
    std::int64_t first_drop = -1;  // the first call where the length fell
    std::int64_t previous = 0;
    const std::vector<std::int32_t> out =
        gen.generate(kPrompt, kNew, [&](std::int32_t) {
            const std::int64_t len = gen.cache().length();
            max_length = std::max(max_length, len);
            if (len < previous && first_drop < 0) {
                first_drop = calls;
            }
            previous = len;
            ++calls;
        });
    TITAN_EXPECT_EQ(calls, kNew);
    TITAN_EXPECT_EQ(out.size(), kPrompt.size() + std::size_t{kNew});
    // kv_write throws for a position >= max_len, so finishing is the position
    // evidence; the cache length never exceeds the context either.
    TITAN_EXPECT(max_length <= kContext);
    TITAN_EXPECT_EQ(max_length, kContext);
    // The prompt (5) fills positions 0..4; new tokens 0..11 fill the cache to
    // 16, so the 13th new token is the first after a slide.
    const std::size_t before_slide = kContext - kPrompt.size() + 1;
    TITAN_EXPECT_EQ(first_drop, static_cast<std::int64_t>(before_slide));
    TITAN_EXPECT(
        std::vector<std::int32_t>(
            out.begin(), out.begin() + kPrompt.size() + before_slide) ==
        std::vector<std::int32_t>(
            naive.begin(), naive.begin() + kPrompt.size() + before_slide));
}

TITAN_TEST(Generator, LongPromptKeepsTail) {
    constexpr std::int64_t kContext = 16;
    Model m(config_with_context(kContext));
    init_model(m, 74);
    Rng data(75);
    std::vector<std::int32_t> prompt(40);
    for (std::int32_t& id : prompt) {
        id = static_cast<std::int32_t>(data.randint(kVocab));
    }
    const std::vector<std::int32_t> tail(prompt.end() - kContext, prompt.end());
    Generator gen(m, SamplerConfig{0.0f, 0, 1.0f}, 0);
    const std::vector<std::int32_t> out = gen.generate(prompt, 1);
    const std::vector<std::int32_t> from_tail = gen.generate(tail, 1);
    TITAN_EXPECT_EQ(out.size(), std::size_t{41});
    TITAN_EXPECT(std::vector<std::int32_t>(out.begin(), out.end() - 1) ==
                 prompt);
    TITAN_EXPECT_EQ(out.back(), from_tail.back());
}

TITAN_TEST(Generator, RejectsEmptyPrompt) {
    Model m(config_with_context(16));
    init_model(m, 76);
    Generator gen(m, SamplerConfig{}, 1);
    TITAN_EXPECT_THROWS(gen.generate({}, 4), titan::Error);
    TITAN_EXPECT_THROWS(gen.generate({}, 0), titan::Error);
    TITAN_EXPECT_THROWS(gen.generate(kPrompt, -1), titan::Error);
    TITAN_EXPECT_THROWS(Generator(m, SamplerConfig{-1.0f, 0, 1.0f}, 1),
                        titan::Error);
    TITAN_EXPECT_THROWS(Generator(m, SamplerConfig{1.0f, -1, 1.0f}, 1),
                        titan::Error);
    TITAN_EXPECT_THROWS(Generator(m, SamplerConfig{1.0f, 0, 0.0f}, 1),
                        titan::Error);
}
