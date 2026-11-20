#include "nn/mlp.hpp"

#include <utility>

#include "ops/elementwise.hpp"

namespace titan::nn {

SwiGLU::SwiGLU(std::string name, std::int64_t dim, std::int64_t hidden)
    : name_(name),
      gate_proj_(name + ".gate_proj", dim, hidden, false),
      up_proj_(name + ".up_proj", dim, hidden, false),
      down_proj_(name + ".down_proj", hidden, dim, false) {}

Tensor SwiGLU::forward(const Tensor& x, ForwardContext& ctx) {
    Tensor gate = gate_proj_.forward(x, ctx);
    Tensor up = up_proj_.forward(x, ctx);
    Tensor act = ops::silu(gate);
    Tensor y = down_proj_.forward(ops::mul(act, up), ctx);
    if (ctx.training) {
        gate_ = std::move(gate);
        up_ = std::move(up);
        act_ = std::move(act);
    } else {
        gate_.reset();
        up_.reset();
        act_.reset();
    }
    return y;
}

Tensor SwiGLU::backward(const Tensor& dy) {
    require_training_forward(gate_.has_value(), name_);
    // h = act ⊙ up with act = silu(gate).
    const Tensor dh = down_proj_.backward(dy);
    const Tensor d_act = ops::mul(dh, *up_);
    const Tensor d_up = ops::mul(dh, *act_);
    const Tensor d_gate = ops::silu_backward(*gate_, d_act);
    // x feeds both projections, so its gradient is the sum of both paths.
    return ops::add(gate_proj_.backward(d_gate), up_proj_.backward(d_up));
}

void SwiGLU::parameters(std::vector<Parameter*>& out) {
    gate_proj_.parameters(out);
    up_proj_.parameters(out);
    down_proj_.parameters(out);
}

GeluMLP::GeluMLP(std::string name, std::int64_t dim, std::int64_t hidden)
    : name_(name),
      fc_(name + ".fc", dim, hidden, true),
      proj_(name + ".proj", hidden, dim, true) {}

Tensor GeluMLP::forward(const Tensor& x, ForwardContext& ctx) {
    Tensor pre = fc_.forward(x, ctx);
    Tensor y = proj_.forward(ops::gelu(pre), ctx);
    if (ctx.training) {
        pre_ = std::move(pre);
    } else {
        pre_.reset();
    }
    return y;
}

Tensor GeluMLP::backward(const Tensor& dy) {
    require_training_forward(pre_.has_value(), name_);
    const Tensor dh = proj_.backward(dy);
    return fc_.backward(ops::gelu_backward(*pre_, dh));
}

void GeluMLP::parameters(std::vector<Parameter*>& out) {
    fc_.parameters(out);
    proj_.parameters(out);
}

}  // namespace titan::nn
