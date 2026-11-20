#include "nn/module.hpp"

#include <utility>

#include "core/error.hpp"

namespace titan::nn {

Parameter make_parameter(std::string name, Shape shape, Device device) {
    Tensor value = Tensor::zeros(shape, device);
    Tensor grad = Tensor::zeros(std::move(shape), device);
    return Parameter{std::move(name), std::move(value), std::move(grad)};
}

void zero_grad(const std::vector<Parameter*>& params) {
    for (Parameter* p : params) {
        p->grad.zero_();
    }
}

void require_training_forward(bool has_cache, const std::string& module_name) {
    TITAN_CHECK(has_cache, module_name +
                               ": backward called without a training-mode "
                               "forward");
}

}  // namespace titan::nn
