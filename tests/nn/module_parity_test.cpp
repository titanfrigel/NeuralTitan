// CPU-vs-CUDA parity of the nn modules: forward and backward of the same
// module, with the same parameters and input, on both devices. Gradient
// checks stay CPU-only; these tests show the CUDA path computes the same
// thing.

#include <cstdint>
#include <source_location>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/attention.hpp"
#include "nn/block.hpp"
#include "nn/linear.hpp"
#include "nn/mlp.hpp"
#include "nn/module.hpp"
#include "nn/norm.hpp"
#include "support/module_gradcheck.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Rng;
using titan::Tensor;
using titan::nn::BlockSpec;
using titan::nn::ForwardContext;
using titan::nn::Module;
using titan::nn::Parameter;
using titan::nn::PositionEncoding;
using titan::test::kParityAtol;
using titan::test::kParityRtol;
using titan::test::random_tensor;

namespace {

constexpr float kEps = 1e-5f;
constexpr float kTheta = 10000.0f;

// Looser bounds for modules that chain several ops (attention, the MLPs, the
// block): each op's float rounding differs between the devices (reduction
// order, cuBLAS blocking), and the differences compound through the chain
// and again through the backward pass.
constexpr double kChainAtol = 1e-4;
constexpr double kChainRtol = 1e-3;

std::vector<Parameter*> params_of(Module& m) {
    std::vector<Parameter*> out;
    m.parameters(out);
    return out;
}

void expect_parity(const std::string& what, const Tensor& cpu,
                   const Tensor& cuda, double atol, double rtol,
                   std::source_location where) {
    const titan::test::ParityResult r =
        titan::test::check_parity(cpu, cuda, atol, rtol);
    if (!r.ok) {
        titan::test::report_failure(where.file_name(),
                                    static_cast<int>(where.line()),
                                    "parity " + what + ": " + r.detail);
    }
}

// `cpu` and `gpu` are two modules built with the same arguments. Draws the
// CPU module's parameters from N(0, 0.5²) with `seed`, gives the GPU module
// copies of them on CUDA (with zero CUDA gradients), runs one training
// forward and one backward on each with the same input x (B, T, dim) and the
// same upstream gradient, and compares y, dx and every parameter gradient.
void expect_module_parity(
    Module& cpu, Module& gpu, const titan::Shape& x_shape, std::uint64_t seed,
    double atol, double rtol,
    std::source_location where = std::source_location::current()) {
    Rng rng(seed);
    titan::test::randomize_parameters(cpu, rng, 0.5f);
    const std::vector<Parameter*> cpu_params = params_of(cpu);
    const std::vector<Parameter*> gpu_params = params_of(gpu);
    TITAN_EXPECT_EQ(cpu_params.size(), gpu_params.size());
    for (std::size_t i = 0; i < cpu_params.size(); ++i) {
        gpu_params[i]->value = cpu_params[i]->value.to(Device::CUDA);
        gpu_params[i]->grad = gpu_params[i]->grad.to(Device::CUDA);
    }

    const Tensor x = random_tensor(x_shape, seed + 1);
    ForwardContext cpu_ctx;
    ForwardContext gpu_ctx;
    const Tensor y_cpu = cpu.forward(x, cpu_ctx);
    const Tensor y_gpu = gpu.forward(x.to(Device::CUDA), gpu_ctx);
    expect_parity("y", y_cpu, y_gpu, atol, rtol, where);

    const Tensor dy = random_tensor(y_cpu.shape(), seed + 2);
    const Tensor dx_cpu = cpu.backward(dy);
    const Tensor dx_gpu = gpu.backward(dy.to(Device::CUDA));
    expect_parity("dx", dx_cpu, dx_gpu, atol, rtol, where);
    for (std::size_t i = 0; i < cpu_params.size(); ++i) {
        expect_parity(cpu_params[i]->name + ".grad", cpu_params[i]->grad,
                      gpu_params[i]->grad, atol, rtol, where);
    }
}

}  // namespace

// Shapes are deliberately not multiples of 16 or 32.

TITAN_TEST(ModuleParity, Linear) {
    TITAN_REQUIRE_CUDA();
    titan::nn::Linear cpu("fc", 37, 29, true);
    titan::nn::Linear gpu("fc", 37, 29, true);
    expect_module_parity(cpu, gpu, {3, 5, 37}, 900, kParityAtol, kParityRtol);
}

TITAN_TEST(ModuleParity, RMSNorm) {
    TITAN_REQUIRE_CUDA();
    titan::nn::RMSNorm cpu("norm", 37, kEps);
    titan::nn::RMSNorm gpu("norm", 37, kEps);
    expect_module_parity(cpu, gpu, {3, 5, 37}, 910, kParityAtol, kParityRtol);
}

TITAN_TEST(ModuleParity, LayerNorm) {
    TITAN_REQUIRE_CUDA();
    titan::nn::LayerNorm cpu("norm", 37, kEps);
    titan::nn::LayerNorm gpu("norm", 37, kEps);
    expect_module_parity(cpu, gpu, {3, 5, 37}, 920, kParityAtol, kParityRtol);
}

TITAN_TEST(ModuleParity, SwiGLU) {
    TITAN_REQUIRE_CUDA();
    titan::nn::SwiGLU cpu("mlp", 24, 44);
    titan::nn::SwiGLU gpu("mlp", 24, 44);
    expect_module_parity(cpu, gpu, {2, 7, 24}, 930, kChainAtol, kChainRtol);
}

TITAN_TEST(ModuleParity, GeluMLP) {
    TITAN_REQUIRE_CUDA();
    titan::nn::GeluMLP cpu("mlp", 24, 44);
    titan::nn::GeluMLP gpu("mlp", 24, 44);
    expect_module_parity(cpu, gpu, {2, 7, 24}, 940, kChainAtol, kChainRtol);
}

TITAN_TEST(ModuleParity, AttentionGqa) {
    TITAN_REQUIRE_CUDA();
    // 4 query heads share 2 KV heads, hd = 6, with RoPE.
    titan::nn::Attention cpu("attn", 24, 4, 2, PositionEncoding::Rope, kTheta);
    titan::nn::Attention gpu("attn", 24, 4, 2, PositionEncoding::Rope, kTheta);
    expect_module_parity(cpu, gpu, {2, 7, 24}, 950, kChainAtol, kChainRtol);
}

TITAN_TEST(ModuleParity, TransformerBlock) {
    TITAN_REQUIRE_CUDA();
    const BlockSpec llama{.n_heads = 4,
                          .n_kv_heads = 2,
                          .mlp_hidden = 44,
                          .mlp = titan::nn::MlpType::SwiGLU,
                          .norm = titan::nn::NormType::RMSNorm,
                          .position = PositionEncoding::Rope};
    titan::nn::TransformerBlock cpu("block", 24, llama, kEps, kTheta);
    titan::nn::TransformerBlock gpu("block", 24, llama, kEps, kTheta);
    expect_module_parity(cpu, gpu, {2, 7, 24}, 960, kChainAtol, kChainRtol);

    const BlockSpec gpt{.n_heads = 4,
                        .n_kv_heads = 4,
                        .mlp_hidden = 44,
                        .mlp = titan::nn::MlpType::Gelu,
                        .norm = titan::nn::NormType::LayerNorm,
                        .position = PositionEncoding::None};
    titan::nn::TransformerBlock cpu_gpt("block", 24, gpt, kEps, kTheta);
    titan::nn::TransformerBlock gpu_gpt("block", 24, gpt, kEps, kTheta);
    expect_module_parity(cpu_gpt, gpu_gpt, {2, 7, 24}, 970, kChainAtol,
                         kChainRtol);
}
