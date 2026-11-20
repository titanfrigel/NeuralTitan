#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "nn/module.hpp"

namespace titan::nn {

// Token-id lookup table "<name>.weight" (vocab, dim). Not a Module: its input
// is integer ids, not a tensor. forward() returns a tensor of shape
// id_shape + {dim}. See docs/math/embedding.md.
class Embedding {
public:
    Embedding(std::string name, std::int64_t vocab, std::int64_t dim);

    Tensor forward(std::span<const std::int32_t> ids, Shape id_shape,
                   ForwardContext& ctx);
    // Scatter-adds dy into the weight gradient.
    void backward(const Tensor& dy);
    void parameters(std::vector<Parameter*>& out) { out.push_back(&weight_); }

    Parameter& weight() { return weight_; }

private:
    std::string name_;
    Parameter weight_;
    std::vector<std::int32_t> ids_;
    bool has_cache_ = false;
};

}  // namespace titan::nn
