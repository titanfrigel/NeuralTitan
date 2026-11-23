// The KV cache: its bookkeeping, and cached inference through attention and
// the model. A cached prefill and every cached decode step must give the logits
// of an uncached forward over the same prefix, on the CPU, on CUDA and with
// blocks of different KV shapes.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "nn/attention.hpp"
#include "nn/kv_cache.hpp"
#include "nn/module.hpp"
#include "support/gradcheck.hpp"
#include "support/module_gradcheck.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Rng;
using titan::Tensor;
using titan::model::llama_style;
using titan::model::Model;
using titan::model::ModelConfig;
using titan::nn::Attention;
using titan::nn::ForwardContext;
using titan::nn::KvCache;
using titan::nn::KvLayerShape;
using titan::nn::PositionEncoding;

namespace {

constexpr double kAtol = 1e-5;
constexpr std::int64_t kVocab = 65;
constexpr std::int64_t kBatch = 2;
constexpr std::int64_t kLen = 10;

// The M1 model-test config: vocab 65, dim 16, 2 layers, 2 heads (hd 8),
// 1 KV head, context 32.
ModelConfig tiny_config() { return llama_style(kVocab, 16, 2, 2, 1, 32); }

// Block 0 keeps 2 heads of size 8 over 1 KV head; block 1 has 4 heads of size
// 4 over 2 KV heads, so the two cache layers differ in both dimensions.
ModelConfig heterogeneous_config() {
    ModelConfig c = tiny_config();
    c.blocks[1].n_heads = 4;
    c.blocks[1].n_kv_heads = 2;
    return c;
}

// Runs `fn`, expecting a titan::Error whose message contains `needle`, so a
// test pins the check that fired rather than any later one.
template <typename Fn>
bool throws_containing(Fn&& fn, const std::string& needle) {
    try {
        fn();
    } catch (const titan::Error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

void init_model(Model& m, std::uint64_t seed) {
    Rng rng(seed);
    m.init(rng);
}

std::vector<std::int32_t> random_ids(std::size_t n, Rng& rng) {
    std::vector<std::int32_t> ids(n);
    for (std::int32_t& id : ids) {
        id = static_cast<std::int32_t>(rng.randint(kVocab));
    }
    return ids;
}

// Tokens first..first+n-1 of each row of the (B, T) id matrix, as (B, n).
std::vector<std::int32_t> slice_ids(const std::vector<std::int32_t>& ids,
                                    std::int64_t B, std::int64_t T,
                                    std::int64_t first, std::int64_t n) {
    std::vector<std::int32_t> out;
    for (std::int64_t b = 0; b < B; ++b) {
        const auto begin = ids.begin() + b * T + first;
        out.insert(out.end(), begin, begin + n);
    }
    return out;
}

// The last position of each row of logits (B, T, V), as a flat (B, V) vector.
std::vector<float> last_rows(const Tensor& logits) {
    const std::vector<float> all = logits.to_vector();
    const std::int64_t B = logits.dim(0);
    const std::int64_t T = logits.dim(1);
    const std::int64_t V = logits.dim(2);
    std::vector<float> out;
    for (std::int64_t b = 0; b < B; ++b) {
        const auto begin = all.begin() + (b * T + T - 1) * V;
        out.insert(out.end(), begin, begin + V);
    }
    return out;
}

void expect_close(const std::vector<float>& got, const std::vector<float>& want,
                  double atol, const std::string& what) {
    TITAN_EXPECT_EQ(got.size(), want.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(got[i]) -
                                         static_cast<double>(want[i])));
    }
    if (!(worst <= atol)) {
        titan::test::report_failure(
            __FILE__, __LINE__,
            what + ": max |got - want| = " + std::to_string(worst));
    }
}

// Runs a cached prefill of the first `prefill` tokens of each row, then
// decodes the remaining T - prefill tokens one at a time. Returns the logits
// of the prefill followed by those of each decode step.
std::vector<Tensor> run_cached(Model& m, const std::vector<std::int32_t>& ids,
                               std::int64_t B, std::int64_t T,
                               std::int64_t prefill) {
    KvCache cache = m.make_kv_cache(B);
    ForwardContext ctx{.training = false, .pos_offset = 0, .kv_cache = &cache};
    std::vector<Tensor> logits;
    logits.push_back(
        m.forward(slice_ids(ids, B, T, 0, prefill), B, prefill, ctx));
    cache.set_length(prefill);
    for (std::int64_t t = prefill; t < T; ++t) {
        ctx.pos_offset = cache.length();
        logits.push_back(m.forward(slice_ids(ids, B, T, t, 1), B, 1, ctx));
        cache.set_length(t + 1);
    }
    return logits;
}

// Checks each decode step of run_cached against the last row of an uncached
// forward over the whole prefix up to and including that step's token.
void expect_decode_matches_full(Model& m, std::uint64_t data_seed) {
    Rng data(data_seed);
    const std::vector<std::int32_t> ids =
        random_ids(static_cast<std::size_t>(kBatch * kLen), data);
    const std::int64_t prefill = 6;
    const std::vector<Tensor> cached =
        run_cached(m, ids, kBatch, kLen, prefill);
    TITAN_EXPECT_EQ(cached.size(),
                    static_cast<std::size_t>(kLen - prefill + 1));
    for (std::int64_t t = prefill; t < kLen; ++t) {
        const Tensor& step = cached[static_cast<std::size_t>(t - prefill + 1)];
        TITAN_EXPECT_EQ(step.dim(1), std::int64_t{1});
        ForwardContext full{.training = false, .pos_offset = 0};
        const Tensor want = m.forward(slice_ids(ids, kBatch, kLen, 0, t + 1),
                                      kBatch, t + 1, full);
        expect_close(step.to_vector(), last_rows(want), kAtol,
                     "decode step at position " + std::to_string(t));
    }
}

}  // namespace

TITAN_TEST(KvCache, ShapesAndLength) {
    KvCache cache({KvLayerShape{.n_kv_heads = 1, .head_dim = 8},
                   KvLayerShape{.n_kv_heads = 3, .head_dim = 4}},
                  2, 5, Device::CPU);
    TITAN_EXPECT_EQ(cache.n_layers(), std::int64_t{2});
    TITAN_EXPECT_EQ(cache.batch(), std::int64_t{2});
    TITAN_EXPECT_EQ(cache.max_len(), std::int64_t{5});
    TITAN_EXPECT_EQ(cache.length(), std::int64_t{0});
    TITAN_EXPECT(cache.keys(0).shape() == (titan::Shape{2, 1, 5, 8}));
    TITAN_EXPECT(cache.values(1).shape() == (titan::Shape{2, 3, 5, 4}));
    for (const float v : cache.keys(1).to_vector()) {
        TITAN_EXPECT_EQ(v, 0.0f);
    }
    cache.set_length(5);
    TITAN_EXPECT_EQ(cache.length(), std::int64_t{5});
    cache.clear();
    TITAN_EXPECT_EQ(cache.length(), std::int64_t{0});
    TITAN_EXPECT_THROWS(cache.set_length(6), titan::Error);
    TITAN_EXPECT_THROWS(cache.set_length(-1), titan::Error);
    TITAN_EXPECT_THROWS(cache.keys(2), titan::Error);
    TITAN_EXPECT_THROWS(cache.values(-1), titan::Error);
    TITAN_EXPECT_THROWS(KvCache({}, 1, 4, Device::CPU), titan::Error);
    TITAN_EXPECT_THROWS(KvCache({KvLayerShape{1, 8}}, 0, 4, Device::CPU),
                        titan::Error);
    TITAN_EXPECT_THROWS(KvCache({KvLayerShape{0, 8}}, 1, 4, Device::CPU),
                        titan::Error);
}

TITAN_TEST(KvCache, PrefillMatchesFullForward) {
    Model m(tiny_config());
    init_model(m, 50);
    Rng data(51);
    const std::vector<std::int32_t> ids =
        random_ids(static_cast<std::size_t>(kBatch * kLen), data);

    ForwardContext full{.training = false, .pos_offset = 0};
    const Tensor want = m.forward(ids, kBatch, kLen, full);

    KvCache cache = m.make_kv_cache(kBatch);
    ForwardContext ctx{.training = false, .pos_offset = 0, .kv_cache = &cache};
    const Tensor got = m.forward(ids, kBatch, kLen, ctx);
    TITAN_EXPECT(got.shape() == want.shape());
    expect_close(got.to_vector(), want.to_vector(), kAtol, "prefill");
}

TITAN_TEST(KvCache, DecodeMatchesFullForward) {
    Model m(tiny_config());
    init_model(m, 52);
    expect_decode_matches_full(m, 53);
}

TITAN_TEST(KvCache, HeterogeneousBlocksDecodeMatchesFullForward) {
    Model m(heterogeneous_config());
    init_model(m, 54);
    const KvCache cache = m.make_kv_cache(kBatch);
    TITAN_EXPECT_EQ(cache.n_layers(), std::int64_t{2});
    TITAN_EXPECT_EQ(cache.max_len(), std::int64_t{32});
    TITAN_EXPECT_EQ(cache.layer_shape(0).n_kv_heads, std::int64_t{1});
    TITAN_EXPECT_EQ(cache.layer_shape(0).head_dim, std::int64_t{8});
    TITAN_EXPECT_EQ(cache.layer_shape(1).n_kv_heads, std::int64_t{2});
    TITAN_EXPECT_EQ(cache.layer_shape(1).head_dim, std::int64_t{4});
    expect_decode_matches_full(m, 55);
}

TITAN_TEST(KvCache, CudaDecodeMatchesCpu) {
    TITAN_REQUIRE_CUDA();
    Model cpu(tiny_config());
    Model gpu(tiny_config());
    init_model(cpu, 56);
    init_model(gpu, 56);
    gpu.to(Device::CUDA);
    Rng data(57);
    const std::vector<std::int32_t> ids =
        random_ids(static_cast<std::size_t>(kBatch * kLen), data);

    const std::vector<Tensor> want = run_cached(cpu, ids, kBatch, kLen, 6);
    const std::vector<Tensor> got = run_cached(gpu, ids, kBatch, kLen, 6);
    TITAN_EXPECT_EQ(got.size(), want.size());
    // The whole model chains dozens of ops that round differently on the two
    // devices, so the bounds are those of the uncached model parity test.
    constexpr double kModelAtol = 1e-4;
    constexpr double kModelRtol = 1e-3;
    for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        TITAN_EXPECT(got[i].device() == Device::CUDA);
        TITAN_EXPECT_PARITY(want[i], got[i], kModelAtol, kModelRtol);
    }
}

TITAN_TEST(KvCache, RejectsTrainingMode) {
    Model m(tiny_config());
    KvCache cache = m.make_kv_cache(1);
    ForwardContext ctx{.training = true, .pos_offset = 0, .kv_cache = &cache};
    const std::vector<std::int32_t> ids{1, 2, 3};
    TITAN_EXPECT(throws_containing([&] { m.forward(ids, 1, 3, ctx); },
                                   "the KV cache is inference-only"));
}

TITAN_TEST(KvCache, RejectsOffsetMismatch) {
    Model m(tiny_config());
    KvCache cache = m.make_kv_cache(1);
    const std::vector<std::int32_t> ids{1, 2, 3};
    // An empty cache needs pos_offset 0.
    ForwardContext ctx{.training = false, .pos_offset = 1, .kv_cache = &cache};
    const std::string mismatch = "must equal the KV cache length";
    TITAN_EXPECT(
        throws_containing([&] { m.forward(ids, 1, 3, ctx); }, mismatch));
    // After a prefill of 3 the next token must start at 3, not 2 or 4.
    ctx.pos_offset = 0;
    m.forward(ids, 1, 3, ctx);
    cache.set_length(3);
    const std::vector<std::int32_t> next{4};
    ctx.pos_offset = 2;
    TITAN_EXPECT(
        throws_containing([&] { m.forward(next, 1, 1, ctx); }, mismatch));
    ctx.pos_offset = 4;
    TITAN_EXPECT(
        throws_containing([&] { m.forward(next, 1, 1, ctx); }, mismatch));
    ctx.pos_offset = 3;
    m.forward(next, 1, 1, ctx);
}

TITAN_TEST(KvCache, RejectsMismatchedCache) {
    Model m(tiny_config());
    const std::vector<std::int32_t> ids{1, 2, 3};
    // One layer for a two-block model.
    KvCache one_layer({KvLayerShape{1, 8}}, 1, 32, Device::CPU);
    ForwardContext ctx{
        .training = false, .pos_offset = 0, .kv_cache = &one_layer};
    TITAN_EXPECT(throws_containing([&] { m.forward(ids, 1, 3, ctx); },
                                   "1 layers for 2 blocks"));
    // The right layer count with the wrong KV head count.
    KvCache wrong_heads({KvLayerShape{2, 8}, KvLayerShape{2, 8}}, 1, 32,
                        Device::CPU);
    ctx.kv_cache = &wrong_heads;
    TITAN_EXPECT(throws_containing([&] { m.forward(ids, 1, 3, ctx); },
                                   "KV cache layer 0 holds 2 heads"));
    // A cache for two sequences with one sequence of input.
    KvCache two_rows = m.make_kv_cache(2);
    ctx.kv_cache = &two_rows;
    TITAN_EXPECT(throws_containing([&] { m.forward(ids, 1, 3, ctx); },
                                   "does not match the KV cache batch 2"));
}

TITAN_TEST(KvCache, RejectsGrowingPastContext) {
    Model m(tiny_config());
    KvCache cache = m.make_kv_cache(1);
    const std::vector<std::int32_t> ids(32, 1);
    ForwardContext ctx{.training = false, .pos_offset = 0, .kv_cache = &cache};
    m.forward(ids, 1, 32, ctx);
    cache.set_length(32);
    ctx.pos_offset = 32;
    const std::vector<std::int32_t> next{1};
    TITAN_EXPECT(throws_containing([&] { m.forward(next, 1, 1, ctx); },
                                   "exceed context_length 32"));
}

TITAN_TEST(KvCache, InferenceCachesNothingForBackward) {
    Rng rng(58);
    Attention attn("attn", 8, 2, 1, PositionEncoding::Rope, 10000.0f);
    titan::test::randomize_parameters(attn, rng, 0.5f);
    const Tensor x = titan::test::random_tensor({1, 3, 8}, rng);
    // A training forward fills the backward state; a cached inference forward
    // must clear it, so attention's own guard (not that of o_proj, which runs
    // after it) rejects backward afterwards.
    ForwardContext train;
    attn.forward(x, train);
    KvCache cache({KvLayerShape{1, 4}}, 1, 8, Device::CPU);
    ForwardContext ctx{.training = false, .pos_offset = 0, .kv_cache = &cache};
    const Tensor y = attn.forward(x, ctx);
    TITAN_EXPECT(throws_containing([&] { attn.backward(y); },
                                   "attn: backward called without"));
}

TITAN_TEST(Attention, PosOffsetWithoutCacheStillThrows) {
    Attention attn("attn", 8, 2, 2, PositionEncoding::Rope, 10000.0f);
    const Tensor x = Tensor::zeros({1, 3, 8}, Device::CPU);
    ForwardContext inference{.training = false, .pos_offset = 1};
    TITAN_EXPECT(throws_containing([&] { attn.forward(x, inference); },
                                   "pos_offset > 0 needs a KV cache"));
    ForwardContext training{.training = true, .pos_offset = 1};
    TITAN_EXPECT(throws_containing([&] { attn.forward(x, training); },
                                   "pos_offset > 0 needs a KV cache"));
}
