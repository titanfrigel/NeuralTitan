#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nn/module.hpp"

namespace titan::nn {

// y = x Wᵀ + b over the last dimension. x has any rank whose last dimension is
// `in`; the leading dimensions are treated as one batch of rows. Parameters:
// "<name>.weight" (out, in) and, with bias, "<name>.bias" (out). Weights start
// at zero: initialize them with init_normal. See docs/math/linear.md.
class Linear : public Module {
public:
    Linear(std::string name, std::int64_t in, std::int64_t out, bool bias,
           Device device = Device::CPU);

    Tensor forward(const Tensor& x, ForwardContext& ctx) override;
    Tensor backward(const Tensor& dy) override;
    void parameters(std::vector<Parameter*>& out) override;

    Parameter& weight() { return weight_; }

private:
    std::string name_;
    std::int64_t in_;
    std::int64_t out_;
    Parameter weight_;
    std::optional<Parameter> bias_;
    std::optional<Tensor> x_;  // input cached by a training-mode forward
};

}  // namespace titan::nn
