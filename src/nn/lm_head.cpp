#include "nn/lm_head.hpp"

#include <utility>

#include "core/error.hpp"
#include "ops/matmul.hpp"

namespace titan::nn {

LMHead::LMHead(Parameter& tied_embedding_weight)
    : name_("lm_head"), weight_(&tied_embedding_weight) {}

LMHead::LMHead(std::string name, std::int64_t dim, std::int64_t vocab)
    : name_(name),
      owned_(make_parameter(name + ".weight", {vocab, dim})),
      weight_(&*owned_) {}

Tensor LMHead::forward(const Tensor& x, ForwardContext& ctx) {
    const std::int64_t dim = weight_->value.dim(1);
    TITAN_CHECK(x.dim(-1) == dim, name_ + ": input " + to_string(x.shape()) +
                                      " does not end in " +
                                      std::to_string(dim));
    const Tensor x2d = x.reshape({x.numel() / dim, dim});
    const Tensor logits2d = ops::matmul(x2d, weight_->value, false, true);
    if (ctx.training) {
        x_ = x;
    } else {
        x_.reset();
    }
    Shape out_shape = x.shape();
    out_shape.back() = weight_->value.dim(0);
    return logits2d.reshape(std::move(out_shape));
}

Tensor LMHead::backward(const Tensor& dy) {
    require_training_forward(x_.has_value(), name_);
    const std::int64_t vocab = weight_->value.dim(0);
    const std::int64_t dim = weight_->value.dim(1);
    Shape expected = x_->shape();
    expected.back() = vocab;
    TITAN_CHECK(dy.shape() == expected,
                name_ + ": dy " + to_string(dy.shape()) + ", expected " +
                    to_string(expected));
    const std::int64_t rows = x_->numel() / dim;
    const Tensor x2d = x_->reshape({rows, dim});
    const Tensor dy2d = dy.reshape({rows, vocab});
    // dE += dyᵀ·x (added to the embedding's gradient when tied), dx = dy·E.
    ops::matmul_acc(weight_->grad, dy2d, x2d, true, false);
    return ops::matmul(dy2d, weight_->value).reshape(x_->shape());
}

void LMHead::parameters(std::vector<Parameter*>& out) {
    if (owned_) {
        out.push_back(&*owned_);
    }
}

}  // namespace titan::nn
