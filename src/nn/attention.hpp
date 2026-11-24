#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nn/linear.hpp"
#include "nn/module.hpp"

namespace titan::nn {

// How attention encodes token positions: rotary embeddings on the queries and
// keys, or nothing (positions then come from elsewhere, e.g. a learned table).
enum class PositionEncoding { Rope, None };

// Causal multi-head self-attention with grouped-query heads (GQA) and optional
// RoPE. Input and output are (B, T, dim). With hd = dim / n_heads, the
// projections are "<name>.q_proj" (dim -> n_heads·hd), "<name>.k_proj" and
// "<name>.v_proj" (dim -> n_kv_heads·hd) and "<name>.o_proj" (dim -> dim), all
// without bias. Each KV head serves n_heads / n_kv_heads query heads; with
// n_kv_heads == n_heads this is ordinary multi-head attention. Query row t
// sees key j when j <= t + ctx.pos_offset, and RoPE uses position
// ctx.pos_offset + t. Without a KV cache forward() rejects pos_offset > 0,
// since there are no earlier keys for the offset to skip over. With
// ctx.kv_cache set (inference only, so training must be false), forward()
// writes the new keys and values into layer ctx.layer of the cache at
// pos_offset, which must equal the cache length, and attends over all
// pos_offset + T cached positions; it caches nothing for backward. See
// docs/math/attention.md and docs/math/kv-cache.md.
class Attention : public Module {
public:
    // Throws unless n_heads >= 1, n_kv_heads >= 1, dim % n_heads == 0,
    // n_heads % n_kv_heads == 0 and, with RoPE, hd is even.
    Attention(std::string name, std::int64_t dim, std::int64_t n_heads,
              std::int64_t n_kv_heads, PositionEncoding pos, float rope_theta);

    Tensor forward(const Tensor& x, ForwardContext& ctx) override;
    Tensor backward(const Tensor& dy) override;
    void parameters(std::vector<Parameter*>& out) override;

private:
    std::string name_;
    std::int64_t dim_;
    std::int64_t n_heads_;
    std::int64_t n_kv_heads_;
    std::int64_t head_dim_;
    PositionEncoding pos_;
    float rope_theta_;
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear o_proj_;
    // Saved by a training-mode forward, all (B, n_heads, T, ·): the queries
    // and the repeated keys after RoPE, the repeated values, the attention
    // probabilities (B, n_heads, T, T), and the position offset.
    std::optional<Tensor> q_;
    std::optional<Tensor> k_rep_;
    std::optional<Tensor> v_rep_;
    std::optional<Tensor> probs_;
    std::int64_t pos_offset_ = 0;
};

}  // namespace titan::nn
