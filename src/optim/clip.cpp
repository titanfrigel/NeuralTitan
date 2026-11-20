#include "optim/clip.hpp"

#include <cmath>

#include "core/error.hpp"
#include "ops/elementwise.hpp"
#include "ops/reduce.hpp"

namespace titan::optim {

double clip_grad_norm(const std::vector<nn::Parameter*>& params,
                      double max_norm) {
    TITAN_CHECK(max_norm > 0.0, "clip_grad_norm: max_norm must be positive");
    double total = 0.0;
    for (const nn::Parameter* p : params) total += ops::sum_squares(p->grad);
    const double norm = std::sqrt(total);
    if (std::isfinite(norm) && norm > max_norm) {
        const float factor = static_cast<float>(max_norm / norm);
        for (nn::Parameter* p : params) ops::scale_(p->grad, factor);
    }
    return norm;
}

}  // namespace titan::optim
