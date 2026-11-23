#include "model/model.hpp"

#include <cmath>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "nn/init.hpp"
#include "nn/norm.hpp"

namespace titan::model {

namespace {

nn::NormType norm_type(const std::string& norm, const std::string& path) {
    if (norm == "rmsnorm") return nn::NormType::RMSNorm;
    if (norm == "layernorm") return nn::NormType::LayerNorm;
    throw Error("Model: " + path +
                " must be \"rmsnorm\" or \"layernorm\", got \"" + norm + "\"");
}

nn::BlockSpec block_spec(const BlockConfig& b, const std::string& path) {
    nn::MlpType mlp;
    if (b.mlp == "swiglu") {
        mlp = nn::MlpType::SwiGLU;
    } else if (b.mlp == "gelu") {
        mlp = nn::MlpType::Gelu;
    } else {
        throw Error("Model: " + path +
                    ".mlp must be \"swiglu\" or \"gelu\", got \"" + b.mlp +
                    "\"");
    }
    nn::PositionEncoding position;
    if (b.position == "rope") {
        position = nn::PositionEncoding::Rope;
    } else if (b.position == "none") {
        position = nn::PositionEncoding::None;
    } else {
        throw Error("Model: " + path +
                    ".position must be \"rope\" or \"none\", got \"" +
                    b.position + "\"");
    }
    return nn::BlockSpec{.n_heads = b.n_heads,
                         .n_kv_heads = b.n_kv_heads,
                         .mlp_hidden = b.mlp_hidden,
                         .mlp = mlp,
                         .norm = norm_type(b.norm, path + ".norm"),
                         .position = position};
}

bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.substr(s.size() - suffix.size()) == suffix;
}

// The projections that write into the residual stream at the end of each
// branch: attention's o_proj and the MLP's down_proj (SwiGLU) or proj (GELU).
bool is_residual_projection(std::string_view name) {
    return ends_with(name, ".o_proj.weight") ||
           ends_with(name, ".down_proj.weight") ||
           ends_with(name, ".mlp.proj.weight");
}

}  // namespace

Model::Model(const ModelConfig& config)
    : config_(config),
      embed_tokens_("embed_tokens", config.vocab_size, config.dim) {
    TITAN_CHECK(!config.blocks.empty(), "Model: the config has no blocks");
    TITAN_CHECK(config.context_length >= 1,
                "Model: context_length must be >= 1, got " +
                    std::to_string(config.context_length));
    for (std::size_t i = 0; i < config.blocks.size(); ++i) {
        const std::string name = "blocks." + std::to_string(i);
        blocks_.push_back(std::make_unique<nn::TransformerBlock>(
            name, config.dim, block_spec(config.blocks[i], name),
            config.norm_eps, config.rope_theta));
    }
    if (norm_type(config.final_norm, "final_norm") == nn::NormType::RMSNorm) {
        final_norm_ = std::make_unique<nn::RMSNorm>("final_norm", config.dim,
                                                    config.norm_eps);
    } else {
        final_norm_ = std::make_unique<nn::LayerNorm>("final_norm", config.dim,
                                                      config.norm_eps);
    }
    if (config.tied_embeddings) {
        lm_head_ = std::make_unique<nn::LMHead>(embed_tokens_.weight());
    } else {
        lm_head_ = std::make_unique<nn::LMHead>("lm_head", config.dim,
                                                config.vocab_size);
    }
}

void Model::init(Rng& rng) {
    constexpr float kStddev = 0.02f;
    const float residual_stddev = static_cast<float>(
        kStddev / std::sqrt(2.0 * static_cast<double>(blocks_.size())));
    for (nn::Parameter* p : parameters()) {
        if (p->value.rank() == 1) {
            // Norm weights start at 1, biases (LayerNorm and Linear) at 0.
            nn::init_constant(*p, ends_with(p->name, ".bias") ? 0.0f : 1.0f);
        } else if (is_residual_projection(p->name)) {
            nn::init_normal(*p, rng, residual_stddev);
        } else {
            nn::init_normal(*p, rng, kStddev);
        }
    }
}

Tensor Model::forward(std::span<const std::int32_t> ids, std::int64_t B,
                      std::int64_t T, nn::ForwardContext& ctx) {
    TITAN_CHECK(B >= 1 && T >= 1, "Model::forward: B and T must be >= 1, got " +
                                      std::to_string(B) + " and " +
                                      std::to_string(T));
    TITAN_CHECK(
        ctx.pos_offset >= 0 && ctx.pos_offset + T <= config_.context_length,
        "Model::forward: positions " + std::to_string(ctx.pos_offset) + ".." +
            std::to_string(ctx.pos_offset + T - 1) + " exceed context_length " +
            std::to_string(config_.context_length));
    TITAN_CHECK(
        ctx.kv_cache == nullptr ||
            ctx.kv_cache->n_layers() ==
                static_cast<std::int64_t>(blocks_.size()),
        "Model::forward: the KV cache has " +
            std::to_string(ctx.kv_cache == nullptr ? 0
                                                   : ctx.kv_cache->n_layers()) +
            " layers for " + std::to_string(blocks_.size()) + " blocks");
    TITAN_CHECK(static_cast<std::int64_t>(ids.size()) == B * T,
                "Model::forward: got " + std::to_string(ids.size()) +
                    " ids for B = " + std::to_string(B) +
                    ", T = " + std::to_string(T));
    Tensor x = embed_tokens_.forward(ids, {B, T}, ctx);  // (B, T, dim)
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        ctx.layer = static_cast<std::int64_t>(i);  // the block's cache layer
        x = blocks_[i]->forward(x, ctx);
    }
    x = final_norm_->forward(x, ctx);
    return lm_head_->forward(x, ctx);  // (B, T, vocab)
}

nn::KvCache Model::make_kv_cache(std::int64_t batch) const {
    std::vector<nn::KvLayerShape> layers;
    layers.reserve(config_.blocks.size());
    for (const BlockConfig& b : config_.blocks) {
        // The constructor checked that n_heads divides dim.
        layers.push_back(nn::KvLayerShape{.n_kv_heads = b.n_kv_heads,
                                          .head_dim = config_.dim / b.n_heads});
    }
    return nn::KvCache(std::move(layers), batch, config_.context_length,
                       device());
}

void Model::backward(const Tensor& dlogits) {
    // The forward chain in reverse. With tied embeddings the head and the
    // embedding both add into embed_tokens.weight's gradient.
    Tensor dx = lm_head_->backward(dlogits);
    dx = final_norm_->backward(dx);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        dx = (*it)->backward(dx);
    }
    embed_tokens_.backward(dx);
}

void Model::to(Device device) {
    // parameters() lists a tied weight once, so it moves once and the head's
    // pointer to that Parameter sees the moved tensors.
    for (nn::Parameter* p : parameters()) {
        p->value = p->value.to(device);
        p->grad = p->grad.to(device);
    }
}

Device Model::device() const { return embed_tokens_.weight().value.device(); }

std::vector<nn::Parameter*> Model::parameters() {
    std::vector<nn::Parameter*> out;
    embed_tokens_.parameters(out);
    for (const auto& block : blocks_) {
        block->parameters(out);
    }
    final_norm_->parameters(out);
    lm_head_->parameters(out);  // empty when tied
    return out;
}

std::int64_t Model::num_parameters() {
    std::int64_t n = 0;
    for (const nn::Parameter* p : parameters()) {
        n += p->value.numel();
    }
    return n;
}

}  // namespace titan::model
