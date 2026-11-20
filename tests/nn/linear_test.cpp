#include <algorithm>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/linear.hpp"
#include "nn/mlp.hpp"
#include "nn/norm.hpp"
#include "support/gradcheck.hpp"
#include "support/module_gradcheck.hpp"

using titan::Rng;
using titan::Tensor;
using titan::nn::ForwardContext;
using titan::nn::Linear;
using titan::test::check_module_gradients;
using titan::test::random_tensor;
using titan::test::randomize_parameters;

namespace {

Tensor make(std::vector<float> v, titan::Shape shape) {
    return Tensor::from_vector(v, std::move(shape));
}

// True when calling backward(dy) on a module that last ran an inference-mode
// forward throws the documented message.
template <typename M>
bool backward_after_inference_throws(M& m, const Tensor& x, const Tensor& dy) {
    ForwardContext ctx;
    ctx.training = false;
    m.forward(x, ctx);
    try {
        m.backward(dy);
    } catch (const titan::Error& e) {
        return std::string(e.what()).find(
                   "backward called without a training-mode forward") !=
               std::string::npos;
    }
    return false;
}

}  // namespace

TITAN_TEST(Linear, Forward) {
    Linear lin("fc", 3, 2, true);
    std::vector<titan::nn::Parameter*> params;
    lin.parameters(params);
    TITAN_EXPECT_EQ(params.size(), 2u);
    TITAN_EXPECT(params[0]->name == "fc.weight");
    TITAN_EXPECT(params[1]->name == "fc.bias");
    TITAN_EXPECT(params[0]->value.shape() == titan::Shape({2, 3}));
    TITAN_EXPECT(params[1]->value.shape() == titan::Shape({2}));
    std::copy_n(std::vector<float>{1, 2, 3, 4, 5, 6}.data(), 6,
                params[0]->value.data());
    std::copy_n(std::vector<float>{10, 20}.data(), 2, params[1]->value.data());

    const Tensor x = make({1, 0, -1, 2, 1, 0}, {2, 3});
    ForwardContext ctx;
    const Tensor y = lin.forward(x, ctx);
    TITAN_EXPECT(y.shape() == titan::Shape({2, 2}));
    TITAN_EXPECT(y.to_vector() == std::vector<float>({8, 18, 14, 33}));
}

TITAN_TEST(Linear, GradCheckNoBias) {
    Rng rng(1);
    Linear lin("fc", 6, 5, false);
    randomize_parameters(lin, rng, 0.5f);
    Tensor x = random_tensor({4, 6}, rng);
    check_module_gradients(lin, x, rng);
}

TITAN_TEST(Linear, GradCheckBias3D) {
    Rng rng(2);
    Linear lin("fc", 4, 3, true);
    randomize_parameters(lin, rng, 0.5f);
    Tensor x = random_tensor({2, 3, 4}, rng);
    check_module_gradients(lin, x, rng);
}

TITAN_TEST(Linear, GradsAccumulate) {
    Rng rng(3);
    Linear lin("fc", 4, 3, true);
    randomize_parameters(lin, rng, 0.5f);
    const Tensor x = random_tensor({5, 4}, rng);
    const Tensor dy = random_tensor({5, 3}, rng);
    std::vector<titan::nn::Parameter*> params;
    lin.parameters(params);

    ForwardContext ctx;
    lin.forward(x, ctx);
    lin.backward(dy);
    std::vector<std::vector<float>> once;
    for (const auto* p : params) {
        once.push_back(p->grad.to_vector());
    }
    lin.backward(dy);
    for (std::size_t k = 0; k < params.size(); ++k) {
        const std::vector<float> twice = params[k]->grad.to_vector();
        for (std::size_t i = 0; i < twice.size(); ++i) {
            TITAN_EXPECT_NEAR(twice[i], 2.0f * once[k][i], 1e-5);
        }
    }
}

TITAN_TEST(Linear, RejectsWrongInputWidth) {
    Linear lin("fc", 4, 3, false);
    ForwardContext ctx;
    const Tensor bad = Tensor::zeros({2, 5}, titan::Device::CPU);
    TITAN_EXPECT_THROWS(lin.forward(bad, ctx), titan::Error);
}

TITAN_TEST(Module, BackwardWithoutTrainingForwardThrows) {
    Linear lin("fc", 4, 4, true);
    titan::nn::RMSNorm rms("n", 4, 1e-5f);
    titan::nn::LayerNorm ln("l", 4, 1e-5f);
    titan::nn::SwiGLU swiglu("s", 4, 6);
    titan::nn::GeluMLP gelu("g", 4, 6);
    const Tensor x = Tensor::zeros({2, 4}, titan::Device::CPU);
    TITAN_EXPECT(backward_after_inference_throws(lin, x, x));
    TITAN_EXPECT(backward_after_inference_throws(rms, x, x));
    TITAN_EXPECT(backward_after_inference_throws(ln, x, x));
    TITAN_EXPECT(backward_after_inference_throws(swiglu, x, x));
    TITAN_EXPECT(backward_after_inference_throws(gelu, x, x));
}
