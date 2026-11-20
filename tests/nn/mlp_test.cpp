#include <string>
#include <vector>

#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/mlp.hpp"
#include "support/gradcheck.hpp"
#include "support/module_gradcheck.hpp"

using titan::Rng;
using titan::Tensor;
using titan::nn::GeluMLP;
using titan::nn::SwiGLU;
using titan::test::check_module_gradients;
using titan::test::random_tensor;
using titan::test::randomize_parameters;

TITAN_TEST(SwiGLU, GradCheck) {
    Rng rng(30);
    SwiGLU mlp("mlp", 6, 10);
    randomize_parameters(mlp, rng, 0.5f);
    Tensor x = random_tensor({2, 3, 6}, rng);
    check_module_gradients(mlp, x, rng);
}

TITAN_TEST(GeluMLP, GradCheck) {
    Rng rng(31);
    GeluMLP mlp("mlp", 6, 10);
    randomize_parameters(mlp, rng, 0.5f);
    Tensor x = random_tensor({2, 3, 6}, rng);
    check_module_gradients(mlp, x, rng);
}

TITAN_TEST(Module, ParameterNames) {
    SwiGLU swiglu("mlp", 4, 6);
    std::vector<titan::nn::Parameter*> params;
    swiglu.parameters(params);
    std::vector<std::string> names;
    for (const auto* p : params) {
        names.push_back(p->name);
    }
    TITAN_EXPECT(names == std::vector<std::string>({"mlp.gate_proj.weight",
                                                    "mlp.up_proj.weight",
                                                    "mlp.down_proj.weight"}));

    GeluMLP gelu("mlp", 4, 6);
    params.clear();
    gelu.parameters(params);
    names.clear();
    for (const auto* p : params) {
        names.push_back(p->name);
    }
    TITAN_EXPECT(names == std::vector<std::string>(
                              {"mlp.fc.weight", "mlp.fc.bias",
                               "mlp.proj.weight", "mlp.proj.bias"}));
}
