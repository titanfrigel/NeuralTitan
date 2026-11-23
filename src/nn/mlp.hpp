#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nn/linear.hpp"
#include "nn/module.hpp"

namespace titan::nn {

// SwiGLU feed-forward block: down(silu(gate(x)) ⊙ up(x)). The projections are
// "<name>.gate_proj", "<name>.up_proj" (dim -> hidden) and "<name>.down_proj"
// (hidden -> dim), all without bias. See docs/math/mlp.md.
class SwiGLU : public Module {
public:
    SwiGLU(std::string name, std::int64_t dim, std::int64_t hidden);

    Tensor forward(const Tensor& x, ForwardContext& ctx) override;
    Tensor backward(const Tensor& dy) override;
    void parameters(std::vector<Parameter*>& out) override;

private:
    std::string name_;
    Linear gate_proj_;
    Linear up_proj_;
    Linear down_proj_;
    // Saved by a training-mode forward: gate(x), up(x) and silu(gate(x)).
    std::optional<Tensor> gate_;
    std::optional<Tensor> up_;
    std::optional<Tensor> act_;
};

// GPT-2 style feed-forward block: proj(gelu(fc(x))). "<name>.fc"
// (dim -> hidden) and "<name>.proj" (hidden -> dim), both with bias.
// See docs/math/mlp.md.
class GeluMLP : public Module {
public:
    GeluMLP(std::string name, std::int64_t dim, std::int64_t hidden);

    Tensor forward(const Tensor& x, ForwardContext& ctx) override;
    Tensor backward(const Tensor& dy) override;
    void parameters(std::vector<Parameter*>& out) override;

private:
    std::string name_;
    Linear fc_;
    Linear proj_;
    // Saved by a training-mode forward: fc(x).
    std::optional<Tensor> pre_;
};

}  // namespace titan::nn
