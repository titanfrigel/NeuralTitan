#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/device.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "model/config.hpp"
#include "nn/block.hpp"
#include "nn/embedding.hpp"
#include "nn/lm_head.hpp"
#include "nn/module.hpp"

namespace titan::model {

// A decoder-only language model built from a ModelConfig:
//   ids -> embed_tokens -> blocks.0 .. blocks.{L-1} -> final_norm -> lm_head
// giving logits (B, T, vocab). With tied embeddings the head reuses
// embed_tokens.weight. Parameter names are "embed_tokens.weight",
// "blocks.N.attn.q_proj.weight", "blocks.N.norm1.weight",
// "final_norm.weight" and, untied only, "lm_head.weight". Leaf names follow
// Hugging Face; the prefixes differ and M5 maps them (blocks.N.attn <->
// model.layers.N.self_attn, norm1/norm2 <-> input_layernorm/
// post_attention_layernorm, final_norm <-> model.norm). See docs/math/model.md.
class Model {
public:
    // Throws titan::Error for an unknown mlp, norm, position or final_norm
    // string, a non-positive size, or an invalid head configuration. The
    // parameters start as the modules build them (weights 0, norm weights 1):
    // call init() before training.
    explicit Model(const ModelConfig& config);
    // The head may point into embed_tokens, so a Model never moves.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Linear and embedding weights ~ N(0, 0.02²); the residual output
    // projections (o_proj, down_proj, proj) use stddev 0.02 / √(2·n_layers);
    // norm weights are 1 and biases 0.
    void init(Rng& rng);

    // ids holds B·T token ids, row-major (B, T). Throws unless B, T >= 1,
    // ids.size() == B·T and T <= context_length. Returns logits (B, T, vocab).
    Tensor forward(std::span<const std::int32_t> ids, std::int64_t B,
                   std::int64_t T, nn::ForwardContext& ctx);
    // Accumulates every parameter gradient from dL/dlogits (B, T, vocab).
    void backward(const Tensor& dlogits);

    // Moves every parameter's value and gradient to `device`, in place: each
    // Parameter object stays where it is and only its tensors are replaced.
    // The tied head refers to the embedding Parameter, so after the move it
    // still shares one storage with embed_tokens.weight. A no-op for tensors
    // already on `device`. Throws titan::Error for Device::CUDA without a GPU.
    // Optimizer state is created on the parameters' device, so construct the
    // optimizer after the move. See docs/math/model.md.
    void to(Device device);
    // The device of the parameters (all of them share one, see to()).
    Device device() const;

    // Every parameter exactly once, in forward order; a tied weight appears
    // once, as embed_tokens.weight.
    std::vector<nn::Parameter*> parameters();
    std::int64_t num_parameters();
    const ModelConfig& config() const { return config_; }
    // The LM head's weight: embed_tokens.weight itself when tied.
    const nn::Parameter& head_weight() const { return lm_head_->weight(); }

private:
    ModelConfig config_;
    nn::Embedding embed_tokens_;
    std::vector<std::unique_ptr<nn::TransformerBlock>> blocks_;
    std::unique_ptr<nn::Module> final_norm_;
    std::unique_ptr<nn::LMHead> lm_head_;
};

}  // namespace titan::model
