#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nn/attention.hpp"
#include "nn/module.hpp"

namespace titan::nn {

enum class MlpType { SwiGLU, Gelu };
enum class NormType { RMSNorm, LayerNorm };

// The shape of one transformer block, the typed form of model::BlockConfig.
struct BlockSpec {
    std::int64_t n_heads;
    std::int64_t n_kv_heads;
    std::int64_t mlp_hidden;
    MlpType mlp;
    NormType norm;
    PositionEncoding position;
};

// A pre-norm transformer block over (B, T, dim):
//   h = x + attn(norm1(x)),  y = h + mlp(norm2(h)).
// Children are "<name>.norm1", "<name>.attn", "<name>.norm2" and
// "<name>.mlp"; the norms and the MLP take the variant chosen by the spec.
// See docs/math/transformer-block.md.
class TransformerBlock : public Module {
public:
    TransformerBlock(std::string name, std::int64_t dim, const BlockSpec& spec,
                     float norm_eps, float rope_theta);

    Tensor forward(const Tensor& x, ForwardContext& ctx) override;
    Tensor backward(const Tensor& dy) override;
    void parameters(std::vector<Parameter*>& out) override;

private:
    std::unique_ptr<Module> norm1_;
    Attention attn_;
    std::unique_ptr<Module> norm2_;
    std::unique_ptr<Module> mlp_;
};

}  // namespace titan::nn
