#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "io/json.hpp"

namespace titan::model {

// One transformer block. mlp is "swiglu" or "gelu", norm is "rmsnorm" or
// "layernorm", position is "rope" or "none".
struct BlockConfig {
    std::int64_t n_heads{}, n_kv_heads{}, mlp_hidden{};
    std::string mlp /*swiglu|gelu*/, norm /*rmsnorm|layernorm*/,
        position /*rope|none*/;
};

// The architecture of a decoder-only model. final_norm is "rmsnorm" or
// "layernorm".
struct ModelConfig {
    std::int64_t vocab_size{}, dim{}, context_length{};
    float rope_theta = 10000.f, norm_eps = 1e-5f;
    bool tied_embeddings = true;
    std::string final_norm = "rmsnorm";
    std::vector<BlockConfig> blocks;
};

// Parses and validates the "model" section of a config file. Required:
// "dim", "context_length", and either a non-empty "blocks" array of block
// objects or "n_layers" plus one "block" object copied n_layers times.
// Optional: "rope_theta", "norm_eps", "tied_embeddings", "final_norm". Every
// block needs "n_heads", "n_kv_heads", "mlp", "mlp_hidden", "norm" and
// "position". vocab_size comes from the tokenizer, not the file.
//
// Throws titan::Error naming the field path, e.g. "model.blocks[2].n_kv_heads",
// for a missing or unknown key, a wrong type, a non-positive size, an unknown
// mlp, norm or position string, dim % n_heads != 0, n_heads % n_kv_heads != 0,
// or an odd head dimension dim / n_heads with RoPE.
ModelConfig parse_model_config(const io::JsonValue& model_section,
                               std::int64_t vocab_size);

// A Llama-style model (FR-7): n_layers identical blocks with SwiGLU of hidden
// size round_up(8·dim/3, 8), RMSNorm, RoPE, an RMSNorm final norm and tied
// embeddings. Throws titan::Error for a non-positive size or an invalid head
// configuration, with the same checks as parse_model_config.
ModelConfig llama_style(std::int64_t vocab, std::int64_t dim,
                        std::int64_t n_layers, std::int64_t n_heads,
                        std::int64_t n_kv_heads, std::int64_t context);

}  // namespace titan::model
