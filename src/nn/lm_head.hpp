#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nn/module.hpp"

namespace titan::nn {

// The output projection from hidden states to vocabulary logits:
// logits = x·Eᵀ over the last dimension, with E of shape (vocab, dim) and no
// bias. x is (..., dim) and the logits are (..., vocab).
//
// Tied mode reuses the token-embedding table as E: the head keeps a reference
// to that Parameter, adds its gradient into the same Parameter::grad, and
// lists no parameters of its own (the embedding lists the shared one). The
// Parameter must outlive the head. Untied mode owns "<name>.weight".
// See docs/math/linear.md and docs/math/model.md.
class LMHead : public Module {
public:
    explicit LMHead(Parameter& tied_embedding_weight);
    LMHead(std::string name, std::int64_t dim, std::int64_t vocab);

    Tensor forward(const Tensor& x, ForwardContext& ctx) override;
    Tensor backward(const Tensor& dy) override;
    void parameters(std::vector<Parameter*>& out) override;

    // The projection weight: the tied embedding Parameter itself in tied
    // mode, the owned "<name>.weight" otherwise.
    const Parameter& weight() const { return *weight_; }

private:
    std::string name_;
    std::optional<Parameter> owned_;  // set in untied mode only
    Parameter* weight_;               // the shared or the owned weight
    std::optional<Tensor> x_;         // input cached by a training forward
};

}  // namespace titan::nn
