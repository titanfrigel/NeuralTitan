#include "support/gradcheck.hpp"

#include <cmath>
#include <cstdint>

#include "core/error.hpp"

namespace titan::test {

namespace {

constexpr float kEps = 1e-2f;
constexpr std::int64_t kMaxChecked = 64;
constexpr double kAbsTol = 1e-3;
constexpr double kRelTol = 2e-2;

// L = sum(forward() * r), accumulated in double.
double loss(const std::function<Tensor()>& forward, const Tensor& r) {
    const Tensor y = forward();
    TITAN_CHECK(y.shape() == r.shape(), "check_gradient: forward output " +
                                            to_string(y.shape()) + " vs r " +
                                            to_string(r.shape()));
    const float* py = y.data();
    const float* pr = r.data();
    double sum = 0.0;
    for (std::int64_t i = 0; i < y.numel(); ++i) {
        sum += static_cast<double>(py[i]) * static_cast<double>(pr[i]);
    }
    return sum;
}

}  // namespace

Tensor random_tensor(Shape s, Rng& rng, float scale) {
    Tensor t = Tensor::empty(std::move(s), Device::CPU);
    rng.fill_normal(t, 0.0f, scale);
    return t;
}

GradCheckResult check_gradient(const std::function<Tensor()>& forward,
                               Tensor& input, const Tensor& analytic_grad,
                               const Tensor& r) {
    TITAN_CHECK(input.shape() == analytic_grad.shape(),
                "check_gradient: input " + to_string(input.shape()) +
                    " vs gradient " + to_string(analytic_grad.shape()));
    const std::int64_t n = input.numel();
    const std::int64_t count = n < kMaxChecked ? n : kMaxChecked;
    float* px = input.data();
    const float* pg = analytic_grad.data();

    for (std::int64_t k = 0; k < count; ++k) {
        const std::int64_t i = n <= kMaxChecked ? k : k * n / kMaxChecked;
        const float original = px[i];
        px[i] = original + kEps;
        const double plus = loss(forward, r);
        px[i] = original - kEps;
        const double minus = loss(forward, r);
        px[i] = original;

        const double numeric = (plus - minus) / (2.0 * kEps);
        const double analytic = pg[i];
        if (std::fabs(analytic - numeric) >
            kAbsTol + kRelTol * std::fabs(numeric)) {
            return {false, "element " + std::to_string(i) + ": analytic " +
                               std::to_string(analytic) + ", numeric " +
                               std::to_string(numeric)};
        }
    }
    return {true, ""};
}

}  // namespace titan::test
