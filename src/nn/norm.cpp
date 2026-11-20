#include "nn/norm.hpp"

#include <utility>

#include "nn/init.hpp"
#include "ops/norm.hpp"

namespace titan::nn {

RMSNorm::RMSNorm(std::string name, std::int64_t dim, float eps)
    : name_(name), eps_(eps), weight_(make_parameter(name + ".weight", {dim})) {
    init_constant(weight_, 1.0f);
}

Tensor RMSNorm::forward(const Tensor& x, ForwardContext& ctx) {
    if (!ctx.training) {
        x_.reset();
        inv_rms_.reset();
        return ops::rms_norm(x, weight_.value, eps_, nullptr);
    }
    Tensor inv_rms = Tensor::zeros({1}, Device::CPU);
    Tensor y = ops::rms_norm(x, weight_.value, eps_, &inv_rms);
    x_ = x;
    inv_rms_ = std::move(inv_rms);
    return y;
}

Tensor RMSNorm::backward(const Tensor& dy) {
    require_training_forward(x_.has_value(), name_);
    return ops::rms_norm_backward(*x_, weight_.value, *inv_rms_, dy,
                                  weight_.grad);
}

void RMSNorm::parameters(std::vector<Parameter*>& out) {
    out.push_back(&weight_);
}

LayerNorm::LayerNorm(std::string name, std::int64_t dim, float eps)
    : name_(name),
      eps_(eps),
      weight_(make_parameter(name + ".weight", {dim})),
      bias_(make_parameter(name + ".bias", {dim})) {
    init_constant(weight_, 1.0f);
}

Tensor LayerNorm::forward(const Tensor& x, ForwardContext& ctx) {
    if (!ctx.training) {
        x_.reset();
        mean_.reset();
        inv_std_.reset();
        return ops::layer_norm(x, weight_.value, bias_.value, eps_, nullptr,
                               nullptr);
    }
    Tensor mean = Tensor::zeros({1}, Device::CPU);
    Tensor inv_std = Tensor::zeros({1}, Device::CPU);
    Tensor y =
        ops::layer_norm(x, weight_.value, bias_.value, eps_, &mean, &inv_std);
    x_ = x;
    mean_ = std::move(mean);
    inv_std_ = std::move(inv_std);
    return y;
}

Tensor LayerNorm::backward(const Tensor& dy) {
    require_training_forward(x_.has_value(), name_);
    return ops::layer_norm_backward(*x_, weight_.value, *mean_, *inv_std_, dy,
                                    weight_.grad, bias_.grad);
}

void LayerNorm::parameters(std::vector<Parameter*>& out) {
    out.push_back(&weight_);
    out.push_back(&bias_);
}

}  // namespace titan::nn
