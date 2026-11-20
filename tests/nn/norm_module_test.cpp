#include <vector>

#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/norm.hpp"
#include "support/gradcheck.hpp"
#include "support/module_gradcheck.hpp"

using titan::Rng;
using titan::Tensor;
using titan::nn::LayerNorm;
using titan::nn::RMSNorm;
using titan::test::check_module_gradients;
using titan::test::random_tensor;
using titan::test::randomize_parameters;

TITAN_TEST(RMSNormModule, InitialParameters) {
    RMSNorm norm("n", 3, 1e-5f);
    TITAN_EXPECT(norm.weight().name == "n.weight");
    TITAN_EXPECT(norm.weight().value.to_vector() ==
                 std::vector<float>({1, 1, 1}));
}

TITAN_TEST(RMSNormModule, GradCheck) {
    Rng rng(20);
    RMSNorm norm("n", 8, 1e-5f);
    randomize_parameters(norm, rng, 1.0f);
    Tensor x = random_tensor({2, 3, 8}, rng);
    check_module_gradients(norm, x, rng);
}

TITAN_TEST(LayerNormModule, InitialParameters) {
    LayerNorm norm("n", 3, 1e-5f);
    TITAN_EXPECT(norm.weight().name == "n.weight");
    TITAN_EXPECT(norm.bias().name == "n.bias");
    TITAN_EXPECT(norm.weight().value.to_vector() ==
                 std::vector<float>({1, 1, 1}));
    TITAN_EXPECT(norm.bias().value.to_vector() ==
                 std::vector<float>({0, 0, 0}));
}

TITAN_TEST(LayerNormModule, GradCheck) {
    Rng rng(21);
    LayerNorm norm("n", 8, 1e-5f);
    randomize_parameters(norm, rng, 1.0f);
    Tensor x = random_tensor({2, 3, 8}, rng);
    check_module_gradients(norm, x, rng);
}
