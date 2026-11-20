#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/tensor.hpp"

namespace titan::nn {

// A learnable tensor and its gradient accumulator. `name` is the dotted path
// of the parameter, for example "blocks.0.mlp.gate_proj.weight".
struct Parameter {
    std::string name;
    Tensor value;
    Tensor grad;
};

// A parameter whose value and gradient are zero-filled.
Parameter make_parameter(std::string name, Shape shape,
                         Device device = Device::CPU);

// Per-call state shared by the modules of one forward pass.
struct ForwardContext {
    // Training mode caches what backward needs; inference mode caches nothing.
    bool training = true;
    // Position of the first token of the input in its sequence.
    std::int64_t pos_offset = 0;
};

// A differentiable layer. forward() in training mode caches what backward()
// needs; backward() returns the gradient with respect to the input and adds
// (+=) the gradients of its parameters to Parameter::grad, so gradients
// accumulate across calls until zero_grad().
//
// A training-mode forward keeps references to its input and intermediate
// tensors (they share storage, nothing is copied). Callers must not modify an
// input in place between forward() and backward().
class Module {
public:
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& x, ForwardContext& ctx) = 0;
    virtual Tensor backward(const Tensor& dy) = 0;
    virtual void parameters(std::vector<Parameter*>& out) = 0;
};

// Sets the gradient of every parameter to zero.
void zero_grad(const std::vector<Parameter*>& params);

// Throws "backward called without a training-mode forward" (naming the
// module) unless `has_cache` is true.
void require_training_forward(bool has_cache, const std::string& module_name);

}  // namespace titan::nn
