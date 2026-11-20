#include "optim/adamw.hpp"

#include <utility>

#include "core/error.hpp"
#include "ops/adamw.hpp"

namespace titan::optim {

AdamW::AdamW(std::vector<nn::Parameter*> params, AdamWConfig config)
    : params_(std::move(params)), config_(config) {
    for (const nn::Parameter* p : params_) {
        TITAN_CHECK(p != nullptr, "AdamW: null parameter");
        m_.push_back(Tensor::zeros(p->value.shape(), p->value.device()));
        v_.push_back(Tensor::zeros(p->value.shape(), p->value.device()));
    }
}

void AdamW::step(float lr) {
    ++steps_;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        nn::Parameter& p = *params_[i];
        const float decay = p.value.rank() == 1 ? 0.0f : config_.weight_decay;
        const ops::AdamWHyper h{lr,          config_.beta1, config_.beta2,
                                config_.eps, decay,         steps_};
        ops::adamw_update(p.value, p.grad, m_[i], v_[i], h);
    }
}

}  // namespace titan::optim
