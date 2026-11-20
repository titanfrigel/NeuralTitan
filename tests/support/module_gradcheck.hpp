#pragma once

#include <source_location>
#include <vector>

#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/init.hpp"
#include "nn/module.hpp"
#include "support/gradcheck.hpp"

namespace titan::test {

// Draws every parameter of `m` from N(0, stddev^2), so that no gradient is
// trivially zero.
inline void randomize_parameters(nn::Module& m, Rng& rng, float stddev) {
    std::vector<nn::Parameter*> params;
    m.parameters(params);
    for (nn::Parameter* p : params) {
        nn::init_normal(*p, rng, stddev);
    }
}

// Gradient check of a module's backward with respect to its input and to
// every parameter, with L = sum(forward(x) * r) and r a fixed random tensor.
// Failures are reported at the caller's location. The upstream gradient is
// therefore r. The parameter gradients come from one backward after zero_grad;
// each parameter is then perturbed in place.
inline void check_module_gradients(
    nn::Module& m, Tensor& x, Rng& rng,
    std::source_location where = std::source_location::current()) {
    const auto forward = [&]() {
        nn::ForwardContext ctx;
        return m.forward(x, ctx);
    };
    const Tensor r = random_tensor(forward().shape(), rng);

    std::vector<nn::Parameter*> params;
    m.parameters(params);
    nn::zero_grad(params);
    const Tensor dx = m.backward(r);
    // m.backward needs a training-mode forward; `forward` above ran one.

    const GradCheckResult input_result = check_gradient(forward, x, dx, r);
    if (!input_result.ok) {
        report_failure(where.file_name(), static_cast<int>(where.line()),
                       "input: " + input_result.detail);
    }
    for (nn::Parameter* p : params) {
        const Tensor analytic = p->grad.clone();
        const GradCheckResult result =
            check_gradient(forward, p->value, analytic, r);
        if (!result.ok) {
            report_failure(where.file_name(), static_cast<int>(where.line()),
                           p->name + ": " + result.detail);
        }
    }
}

}  // namespace titan::test
