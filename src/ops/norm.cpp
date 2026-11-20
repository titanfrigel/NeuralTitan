#include "ops/norm.hpp"

#include <cstdint>
#include <string>

#include "ops/cpu/norm.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

namespace {

std::int64_t row_count(const Tensor& x) { return x.numel() / x.dim(-1); }

// Throws naming the op and the parameter unless t has shape {d}.
void require_vector(const char* op, const char* name, const Tensor& t,
                    std::int64_t d) {
    TITAN_CHECK(t.shape() == Shape{d},
                std::string("ops::") + op + ": " + name + " must have shape " +
                    to_string(Shape{d}) + ", got " + to_string(t.shape()));
}

// Throws unless the saved statistic holds one value per row of x.
void require_stat(const char* op, const char* name, const Tensor& stat,
                  const Tensor& x) {
    TITAN_CHECK(stat.numel() == row_count(x),
                std::string("ops::") + op + ": " + name + " must hold " +
                    std::to_string(row_count(x)) + " values for x " +
                    to_string(x.shape()) + ", got shape " +
                    to_string(stat.shape()));
}

}  // namespace

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps,
                Tensor* inv_rms_out) {
    detail::require_cpu("rms_norm", {&x, &weight});
    require_vector("rms_norm", "weight", weight, x.dim(-1));
    return cpu::rms_norm(x, weight, eps, inv_rms_out);
}

Tensor rms_norm_backward(const Tensor& x, const Tensor& weight,
                         const Tensor& inv_rms, const Tensor& dy,
                         Tensor& dweight_acc) {
    detail::require_cpu("rms_norm_backward",
                        {&x, &weight, &inv_rms, &dy, &dweight_acc});
    detail::require_same_shape("rms_norm_backward", x, dy);
    require_vector("rms_norm_backward", "weight", weight, x.dim(-1));
    require_vector("rms_norm_backward", "dweight_acc", dweight_acc, x.dim(-1));
    require_stat("rms_norm_backward", "inv_rms", inv_rms, x);
    return cpu::rms_norm_backward(x, weight, inv_rms, dy, dweight_acc);
}

Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  float eps, Tensor* mean_out, Tensor* inv_std_out) {
    detail::require_cpu("layer_norm", {&x, &weight, &bias});
    require_vector("layer_norm", "weight", weight, x.dim(-1));
    require_vector("layer_norm", "bias", bias, x.dim(-1));
    return cpu::layer_norm(x, weight, bias, eps, mean_out, inv_std_out);
}

Tensor layer_norm_backward(const Tensor& x, const Tensor& weight,
                           const Tensor& mean, const Tensor& inv_std,
                           const Tensor& dy, Tensor& dweight_acc,
                           Tensor& dbias_acc) {
    detail::require_cpu("layer_norm_backward", {&x, &weight, &mean, &inv_std,
                                                &dy, &dweight_acc, &dbias_acc});
    detail::require_same_shape("layer_norm_backward", x, dy);
    require_vector("layer_norm_backward", "weight", weight, x.dim(-1));
    require_vector("layer_norm_backward", "dweight_acc", dweight_acc,
                   x.dim(-1));
    require_vector("layer_norm_backward", "dbias_acc", dbias_acc, x.dim(-1));
    require_stat("layer_norm_backward", "mean", mean, x);
    require_stat("layer_norm_backward", "inv_std", inv_std, x);
    return cpu::layer_norm_backward(x, weight, mean, inv_std, dy, dweight_acc,
                                    dbias_acc);
}

}  // namespace titan::ops
