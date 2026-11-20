#include "nn/embedding.hpp"

#include <utility>

#include "ops/embedding.hpp"

namespace titan::nn {

Embedding::Embedding(std::string name, std::int64_t vocab, std::int64_t dim)
    : name_(name), weight_(make_parameter(name + ".weight", {vocab, dim})) {}

Tensor Embedding::forward(std::span<const std::int32_t> ids, Shape id_shape,
                          ForwardContext& ctx) {
    Tensor y = ops::embedding(weight_.value, ids, std::move(id_shape));
    if (ctx.training) {
        ids_.assign(ids.begin(), ids.end());
    } else {
        ids_.clear();
    }
    has_cache_ = ctx.training;
    return y;
}

void Embedding::backward(const Tensor& dy) {
    require_training_forward(has_cache_, name_);
    ops::embedding_backward(weight_.grad, ids_, dy);
}

}  // namespace titan::nn
