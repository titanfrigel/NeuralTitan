#include "nn/block.hpp"

#include <utility>

#include "nn/mlp.hpp"
#include "nn/norm.hpp"
#include "ops/elementwise.hpp"

namespace titan::nn {

namespace {

std::unique_ptr<Module> make_norm(NormType type, std::string name,
                                  std::int64_t dim, float eps) {
    if (type == NormType::RMSNorm) {
        return std::make_unique<RMSNorm>(std::move(name), dim, eps);
    }
    return std::make_unique<LayerNorm>(std::move(name), dim, eps);
}

std::unique_ptr<Module> make_mlp(MlpType type, std::string name,
                                 std::int64_t dim, std::int64_t hidden) {
    if (type == MlpType::SwiGLU) {
        return std::make_unique<SwiGLU>(std::move(name), dim, hidden);
    }
    return std::make_unique<GeluMLP>(std::move(name), dim, hidden);
}

}  // namespace

TransformerBlock::TransformerBlock(std::string name, std::int64_t dim,
                                   const BlockSpec& spec, float norm_eps,
                                   float rope_theta)
    : norm1_(make_norm(spec.norm, name + ".norm1", dim, norm_eps)),
      attn_(name + ".attn", dim, spec.n_heads, spec.n_kv_heads, spec.position,
            rope_theta),
      norm2_(make_norm(spec.norm, name + ".norm2", dim, norm_eps)),
      mlp_(make_mlp(spec.mlp, name + ".mlp", dim, spec.mlp_hidden)) {}

Tensor TransformerBlock::forward(const Tensor& x, ForwardContext& ctx) {
    // The residual adds are out of place: the children keep references to
    // their inputs (x, h) until backward.
    const Tensor h = ops::add(x, attn_.forward(norm1_->forward(x, ctx), ctx));
    return ops::add(h, mlp_->forward(norm2_->forward(h, ctx), ctx));
}

Tensor TransformerBlock::backward(const Tensor& dy) {
    // y = h + mlp(norm2(h)): h receives dy directly and through the branch.
    const Tensor dh = ops::add(dy, norm2_->backward(mlp_->backward(dy)));
    // h = x + attn(norm1(x)): likewise for x.
    return ops::add(dh, norm1_->backward(attn_.backward(dh)));
}

void TransformerBlock::parameters(std::vector<Parameter*>& out) {
    norm1_->parameters(out);
    attn_.parameters(out);
    norm2_->parameters(out);
    mlp_->parameters(out);
}

}  // namespace titan::nn
