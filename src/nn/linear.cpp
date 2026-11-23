#include "nn/linear.hpp"

#include <utility>

#include "core/error.hpp"
#include "ops/elementwise.hpp"
#include "ops/matmul.hpp"
#include "ops/reduce.hpp"

namespace titan::nn {

Linear::Linear(std::string name, std::int64_t in, std::int64_t out, bool bias,
               Device device)
    : name_(name),
      in_(in),
      out_(out),
      weight_(make_parameter(name + ".weight", {out, in}, device)) {
    if (bias) {
        bias_ = make_parameter(name + ".bias", {out}, device);
    }
}

Tensor Linear::forward(const Tensor& x, ForwardContext& ctx) {
    TITAN_CHECK(x.dim(-1) == in_, name_ + ": input " + to_string(x.shape()) +
                                      " does not end in " +
                                      std::to_string(in_));
    const std::int64_t rows = x.numel() / in_;
    const Tensor x2d = x.reshape({rows, in_});
    Tensor y2d = ops::matmul(x2d, weight_.value, false, true);
    if (bias_) {
        ops::add_rowwise_(y2d, bias_->value);
    }
    if (ctx.training) {
        x_ = x;
    } else {
        x_.reset();
    }
    Shape out_shape = x.shape();
    out_shape.back() = out_;
    return y2d.reshape(std::move(out_shape));
}

Tensor Linear::backward(const Tensor& dy) {
    require_training_forward(x_.has_value(), name_);
    const Tensor& x = *x_;
    Shape expected = x.shape();
    expected.back() = out_;
    TITAN_CHECK(dy.shape() == expected,
                name_ + ": dy " + to_string(dy.shape()) + ", expected " +
                    to_string(expected));
    const std::int64_t rows = x.numel() / in_;
    const Tensor x2d = x.reshape({rows, in_});
    const Tensor dy2d = dy.reshape({rows, out_});
    // dW += dyᵀ x, db += sum of dy over the rows, dx = dy W.
    ops::matmul_acc(weight_.grad, dy2d, x2d, true, false);
    if (bias_) {
        ops::add_(bias_->grad, ops::sum_leading(dy2d));
    }
    return ops::matmul(dy2d, weight_.value).reshape(x.shape());
}

void Linear::parameters(std::vector<Parameter*>& out) {
    out.push_back(&weight_);
    if (bias_) {
        out.push_back(&*bias_);
    }
}

}  // namespace titan::nn
