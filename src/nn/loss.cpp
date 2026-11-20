#include "nn/loss.hpp"

#include <string>

#include "core/error.hpp"
#include "ops/cross_entropy.hpp"

namespace titan::nn {

float CrossEntropyLoss::forward(const Tensor& logits,
                                std::span<const std::int32_t> targets) {
    TITAN_CHECK(logits.rank() >= 2,
                "CrossEntropyLoss: logits must have rank >= 2 (..., V), got " +
                    to_string(logits.shape()));
    const std::int64_t vocab = logits.dim(-1);
    const std::int64_t rows = logits.numel() / vocab;
    // A placeholder: ops::cross_entropy assigns the (rows, vocab) gradient.
    Tensor dlogits2d = Tensor::empty({1}, Device::CPU);
    // ops::cross_entropy checks the target count and range.
    const float loss =
        ops::cross_entropy(logits.reshape({rows, vocab}), targets, &dlogits2d);
    dlogits_ = dlogits2d.reshape(logits.shape());
    return loss;
}

Tensor CrossEntropyLoss::backward() const {
    TITAN_CHECK(dlogits_.has_value(),
                "CrossEntropyLoss: backward called before forward");
    return *dlogits_;
}

}  // namespace titan::nn
