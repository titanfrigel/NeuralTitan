#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nn/module.hpp"

namespace titan::nn {

// RMSNorm over the last dimension: y = x / rms(x) * weight. Parameter
// "<name>.weight" (dim), initialized to 1. See docs/math/normalization.md.
class RMSNorm : public Module {
public:
    RMSNorm(std::string name, std::int64_t dim, float eps);

    Tensor forward(const Tensor& x, ForwardContext& ctx) override;
    Tensor backward(const Tensor& dy) override;
    void parameters(std::vector<Parameter*>& out) override;

    Parameter& weight() { return weight_; }

private:
    std::string name_;
    float eps_;
    Parameter weight_;
    // Saved by a training-mode forward.
    std::optional<Tensor> x_;
    std::optional<Tensor> inv_rms_;
};

// LayerNorm over the last dimension: y = (x - mean) / std * weight + bias.
// Parameters "<name>.weight" (dim) initialized to 1 and "<name>.bias" (dim)
// initialized to 0. See docs/math/normalization.md.
class LayerNorm : public Module {
public:
    LayerNorm(std::string name, std::int64_t dim, float eps);

    Tensor forward(const Tensor& x, ForwardContext& ctx) override;
    Tensor backward(const Tensor& dy) override;
    void parameters(std::vector<Parameter*>& out) override;

    Parameter& weight() { return weight_; }
    Parameter& bias() { return bias_; }

private:
    std::string name_;
    float eps_;
    Parameter weight_;
    Parameter bias_;
    // Saved by a training-mode forward.
    std::optional<Tensor> x_;
    std::optional<Tensor> mean_;
    std::optional<Tensor> inv_std_;
};

}  // namespace titan::nn
