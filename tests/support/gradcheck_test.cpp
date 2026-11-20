#include <string>
#include <vector>

#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/elementwise.hpp"
#include "support/gradcheck.hpp"

using titan::Rng;
using titan::Tensor;
using titan::test::check_gradient;
using titan::test::random_tensor;

TITAN_TEST(GradCheck, AcceptsCorrectGradient) {
    Rng rng(1);
    Tensor x = random_tensor({2, 3}, rng);
    const Tensor r = random_tensor({2, 3}, rng);
    // y = 3x, so dL/dx = 3r.
    const Tensor grad = titan::ops::scale(r, 3.0f);
    const auto result =
        check_gradient([&] { return titan::ops::scale(x, 3.0f); }, x, grad, r);
    TITAN_EXPECT(result.ok);
}

TITAN_TEST(GradCheck, DetectsWrongGradient) {
    Rng rng(1);
    Tensor x = random_tensor({2, 3}, rng);
    const Tensor r = random_tensor({2, 3}, rng);
    const Tensor doubled = titan::ops::scale(r, 6.0f);
    const auto result = check_gradient(
        [&] { return titan::ops::scale(x, 3.0f); }, x, doubled, r);
    TITAN_EXPECT(!result.ok);
    TITAN_EXPECT(result.detail.find("element") != std::string::npos);
}

TITAN_TEST(GradCheck, RestoresInput) {
    Rng rng(2);
    Tensor x = random_tensor({100}, rng);
    const std::vector<float> before = x.to_vector();
    const Tensor r = random_tensor({100}, rng);
    const Tensor grad = titan::ops::scale(r, 3.0f);
    check_gradient([&] { return titan::ops::scale(x, 3.0f); }, x, grad, r);
    TITAN_EXPECT(x.to_vector() == before);
}
