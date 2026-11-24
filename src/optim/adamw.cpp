#include "optim/adamw.hpp"

#include <cstddef>
#include <string>
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

AdamWState AdamW::state() const {
    // to() shares storage on the same device, so clone the CPU case.
    const auto host_copy = [](const Tensor& t) {
        return t.device() == Device::CPU ? t.clone() : t.to(Device::CPU);
    };
    AdamWState s;
    s.steps = steps_;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        s.m.push_back(host_copy(m_[i]));
        s.v.push_back(host_copy(v_[i]));
    }
    return s;
}

void AdamW::load_state(const AdamWState& s) {
    TITAN_CHECK(s.steps >= 0, "AdamW::load_state: negative step count " +
                                  std::to_string(s.steps));
    TITAN_CHECK(s.m.size() == params_.size() && s.v.size() == params_.size(),
                "AdamW::load_state: state has " + std::to_string(s.m.size()) +
                    " first and " + std::to_string(s.v.size()) +
                    " second moments for " + std::to_string(params_.size()) +
                    " parameters");
    for (std::size_t i = 0; i < params_.size(); ++i) {
        const nn::Parameter& p = *params_[i];
        for (const Tensor* t : {&s.m[i], &s.v[i]}) {
            TITAN_CHECK(t->shape() == p.value.shape(),
                        "AdamW::load_state: parameter " + p.name +
                            " has shape " + to_string(p.value.shape()) +
                            " but its moment has shape " +
                            to_string(t->shape()));
        }
    }
    const auto device_copy = [](const Tensor& t, Device d) {
        return t.device() == d ? t.clone() : t.to(d);
    };
    for (std::size_t i = 0; i < params_.size(); ++i) {
        const Device d = params_[i]->value.device();
        m_[i] = device_copy(s.m[i], d);
        v_[i] = device_copy(s.v[i], d);
    }
    steps_ = s.steps;
}

}  // namespace titan::optim
