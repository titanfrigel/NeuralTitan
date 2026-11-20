#include <cstddef>
#include <cstdint>
#include <source_location>
#include <string>
#include <string_view>

#include "core/error.hpp"
#include "harness/test.hpp"
#include "io/json.hpp"
#include "model/config.hpp"

using titan::io::parse_json;
using titan::model::ModelConfig;
using titan::model::parse_model_config;

namespace {

constexpr std::int64_t kVocab = 65;

ModelConfig parse(std::string_view text) {
    return parse_model_config(parse_json(text), kVocab);
}

// Expects parse(text) to throw a titan::Error whose message contains
// `needle`, and reports a failure naming the actual message otherwise.
void expect_rejects(
    std::string_view text, const std::string& needle,
    std::source_location where = std::source_location::current()) {
    try {
        parse(text);
    } catch (const titan::Error& e) {
        const std::string message = e.what();
        if (message.find(needle) == std::string::npos) {
            titan::test::report_failure(
                where.file_name(), static_cast<int>(where.line()),
                "message lacks \"" + needle + "\": " + message);
        }
        return;
    }
    titan::test::report_failure(where.file_name(),
                                static_cast<int>(where.line()),
                                "did not throw, expected \"" + needle + "\"");
}

}  // namespace

TITAN_TEST(ModelConfig, ParsesBlocksArray) {
    const ModelConfig c = parse(R"({
        "dim": 16, "context_length": 32,
        "rope_theta": 500000, "norm_eps": 1e-6, "tied_embeddings": false,
        "final_norm": "layernorm",
        "blocks": [
            {"n_heads": 4, "n_kv_heads": 2, "mlp": "swiglu", "mlp_hidden": 48,
             "norm": "rmsnorm", "position": "rope"},
            {"n_heads": 2, "n_kv_heads": 2, "mlp": "gelu", "mlp_hidden": 64,
             "norm": "layernorm", "position": "none"}
        ]})");
    TITAN_EXPECT_EQ(c.vocab_size, kVocab);
    TITAN_EXPECT_EQ(c.dim, std::int64_t{16});
    TITAN_EXPECT_EQ(c.context_length, std::int64_t{32});
    TITAN_EXPECT_EQ(c.rope_theta, 500000.0f);
    TITAN_EXPECT_EQ(c.norm_eps, 1e-6f);
    TITAN_EXPECT_EQ(c.tied_embeddings, false);
    TITAN_EXPECT_EQ(c.final_norm, std::string("layernorm"));
    TITAN_EXPECT_EQ(c.blocks.size(), std::size_t{2});
    TITAN_EXPECT_EQ(c.blocks[0].n_heads, std::int64_t{4});
    TITAN_EXPECT_EQ(c.blocks[0].n_kv_heads, std::int64_t{2});
    TITAN_EXPECT_EQ(c.blocks[0].mlp, std::string("swiglu"));
    TITAN_EXPECT_EQ(c.blocks[0].mlp_hidden, std::int64_t{48});
    TITAN_EXPECT_EQ(c.blocks[0].norm, std::string("rmsnorm"));
    TITAN_EXPECT_EQ(c.blocks[0].position, std::string("rope"));
    TITAN_EXPECT_EQ(c.blocks[1].n_heads, std::int64_t{2});
    TITAN_EXPECT_EQ(c.blocks[1].mlp, std::string("gelu"));
    TITAN_EXPECT_EQ(c.blocks[1].mlp_hidden, std::int64_t{64});
    TITAN_EXPECT_EQ(c.blocks[1].norm, std::string("layernorm"));
    TITAN_EXPECT_EQ(c.blocks[1].position, std::string("none"));
}

TITAN_TEST(ModelConfig, ParsesNLayersShorthand) {
    const ModelConfig c = parse(R"({
        "dim": 128, "context_length": 128, "n_layers": 4,
        "block": {"n_heads": 4, "n_kv_heads": 2, "mlp": "swiglu",
                  "mlp_hidden": 344, "norm": "rmsnorm", "position": "rope"}})");
    TITAN_EXPECT_EQ(c.blocks.size(), std::size_t{4});
    for (const auto& b : c.blocks) {
        TITAN_EXPECT_EQ(b.n_heads, std::int64_t{4});
        TITAN_EXPECT_EQ(b.n_kv_heads, std::int64_t{2});
        TITAN_EXPECT_EQ(b.mlp_hidden, std::int64_t{344});
        TITAN_EXPECT_EQ(b.mlp, std::string("swiglu"));
        TITAN_EXPECT_EQ(b.norm, std::string("rmsnorm"));
        TITAN_EXPECT_EQ(b.position, std::string("rope"));
    }
    // Defaults for the optional fields.
    TITAN_EXPECT_EQ(c.rope_theta, 10000.0f);
    TITAN_EXPECT_EQ(c.norm_eps, 1e-5f);
    TITAN_EXPECT_EQ(c.tied_embeddings, true);
    TITAN_EXPECT_EQ(c.final_norm, std::string("rmsnorm"));
}

TITAN_TEST(ModelConfig, RejectsInvalid) {
    // The third block is the bad one, so each path names blocks[2].
    const auto three_blocks = [](std::string_view third) {
        const std::string good =
            R"({"n_heads": 4, "n_kv_heads": 2, "mlp": "swiglu",
                "mlp_hidden": 32, "norm": "rmsnorm", "position": "rope"})";
        return std::string(R"({"dim": 16, "context_length": 32, "blocks": [)") +
               good + "," + good + "," + std::string(third) + "]}";
    };

    // dim % n_heads != 0: 16 % 3.
    expect_rejects(three_blocks(R"({"n_heads": 3, "n_kv_heads": 1,
        "mlp": "swiglu", "mlp_hidden": 32, "norm": "rmsnorm",
        "position": "none"})"),
                   "model.blocks[2].n_heads");
    // n_heads % n_kv_heads != 0: 4 % 3.
    expect_rejects(three_blocks(R"({"n_heads": 4, "n_kv_heads": 3,
        "mlp": "swiglu", "mlp_hidden": 32, "norm": "rmsnorm",
        "position": "rope"})"),
                   "model.blocks[2].n_kv_heads");
    // Odd head dimension with RoPE: 16 / 16 = 1.
    expect_rejects(three_blocks(R"({"n_heads": 16, "n_kv_heads": 4,
        "mlp": "swiglu", "mlp_hidden": 32, "norm": "rmsnorm",
        "position": "rope"})"),
                   "model.blocks[2].position");
    // Unknown mlp, norm and position strings.
    expect_rejects(three_blocks(R"({"n_heads": 4, "n_kv_heads": 2,
        "mlp": "relu", "mlp_hidden": 32, "norm": "rmsnorm",
        "position": "rope"})"),
                   "model.blocks[2].mlp");
    expect_rejects(three_blocks(R"({"n_heads": 4, "n_kv_heads": 2,
        "mlp": "swiglu", "mlp_hidden": 32, "norm": "batchnorm",
        "position": "rope"})"),
                   "model.blocks[2].norm");
    expect_rejects(three_blocks(R"({"n_heads": 4, "n_kv_heads": 2,
        "mlp": "swiglu", "mlp_hidden": 32, "norm": "rmsnorm",
        "position": "alibi"})"),
                   "model.blocks[2].position");
    // The shorthand reports paths under model.block.
    expect_rejects(R"({"dim": 16, "context_length": 32, "n_layers": 2,
        "block": {"n_heads": 4, "n_kv_heads": 3, "mlp": "swiglu",
                  "mlp_hidden": 32, "norm": "rmsnorm", "position": "rope"}})",
                   "model.block.n_kv_heads");
}

TITAN_TEST(ModelConfig, RejectsMalformed) {
    const std::string block =
        R"({"n_heads": 4, "n_kv_heads": 2, "mlp": "swiglu",
            "mlp_hidden": 32, "norm": "rmsnorm", "position": "rope"})";
    const std::string blocks = R"("blocks": [)" + block + "]";

    expect_rejects("[]", "model: expected an object");
    // Missing required fields name their path.
    expect_rejects(R"({"context_length": 32, )" + blocks + "}", "model.dim");
    expect_rejects(R"({"dim": 16, )" + blocks + "}", "model.context_length");
    expect_rejects(R"({"dim": 16, "context_length": 32, "blocks": [
        {"n_heads": 4, "mlp": "swiglu", "mlp_hidden": 32, "norm": "rmsnorm",
         "position": "rope"}]})",
                   "model.blocks[0].n_kv_heads");
    expect_rejects(R"({"dim": 16, "context_length": 32})", "model.blocks");
    expect_rejects(R"({"dim": 16, "context_length": 32, "n_layers": 2})",
                   "model.block");
    // blocks and the shorthand are exclusive.
    expect_rejects(
        R"({"dim": 16, "context_length": 32, "n_layers": 1, )" + blocks + "}",
        "model.n_layers");
    expect_rejects(R"({"dim": 16, "context_length": 32, "blocks": []})",
                   "model.blocks");
    // Unknown keys are rejected, with their path.
    expect_rejects(
        R"({"dim": 16, "context_length": 32, "vocab_size": 9, )" + blocks + "}",
        "model.vocab_size");
    expect_rejects(R"({"dim": 16, "context_length": 32, "blocks": [
        {"n_heads": 4, "n_kv_heads": 2, "mlp": "swiglu", "mlp_hidden": 32,
         "norm": "rmsnorm", "position": "rope", "bias": true}]})",
                   "model.blocks[0].bias");
    // Wrong types and values.
    expect_rejects(R"({"dim": "16", "context_length": 32, )" + blocks + "}",
                   "model.dim");
    expect_rejects(R"({"dim": 16.5, "context_length": 32, )" + blocks + "}",
                   "model.dim");
    expect_rejects(R"({"dim": 0, "context_length": 32, )" + blocks + "}",
                   "model.dim");
    expect_rejects(R"({"dim": 16, "context_length": -1, )" + blocks + "}",
                   "model.context_length");
    expect_rejects(R"({"dim": 16, "context_length": 32, "n_layers": 0,
        "block": )" + block +
                       "}",
                   "model.n_layers");
    expect_rejects(
        R"({"dim": 16, "context_length": 32, "rope_theta": 0, )" + blocks + "}",
        "model.rope_theta");
    expect_rejects(
        R"({"dim": 16, "context_length": 32, "norm_eps": -1, )" + blocks + "}",
        "model.norm_eps");
    expect_rejects(R"({"dim": 16, "context_length": 32,
        "tied_embeddings": 1, )" +
                       blocks + "}",
                   "model.tied_embeddings");
    expect_rejects(R"({"dim": 16, "context_length": 32,
        "final_norm": "none", )" +
                       blocks + "}",
                   "model.final_norm");
    // vocab_size comes from the tokenizer and must be positive.
    TITAN_EXPECT_THROWS(
        parse_model_config(
            parse_json(R"({"dim": 16, "context_length": 32, )" + blocks + "}"),
            0),
        titan::Error);
}

TITAN_TEST(ModelConfig, LlamaStylePreset) {
    const ModelConfig c = titan::model::llama_style(kVocab, 128, 4, 4, 2, 128);
    TITAN_EXPECT_EQ(c.vocab_size, kVocab);
    TITAN_EXPECT_EQ(c.dim, std::int64_t{128});
    TITAN_EXPECT_EQ(c.context_length, std::int64_t{128});
    TITAN_EXPECT_EQ(c.tied_embeddings, true);
    TITAN_EXPECT_EQ(c.final_norm, std::string("rmsnorm"));
    TITAN_EXPECT_EQ(c.blocks.size(), std::size_t{4});
    for (const auto& b : c.blocks) {
        TITAN_EXPECT_EQ(b.n_heads, std::int64_t{4});
        TITAN_EXPECT_EQ(b.n_kv_heads, std::int64_t{2});
        // 8·128/3 = 341.3, rounded up to a multiple of 8.
        TITAN_EXPECT_EQ(b.mlp_hidden, std::int64_t{344});
        TITAN_EXPECT_EQ(b.mlp, std::string("swiglu"));
        TITAN_EXPECT_EQ(b.norm, std::string("rmsnorm"));
        TITAN_EXPECT_EQ(b.position, std::string("rope"));
    }
}
