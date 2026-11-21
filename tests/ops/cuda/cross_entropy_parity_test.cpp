// CPU-vs-CUDA parity of the fused softmax cross-entropy: the mean loss and the
// gradient (softmax - onehot) / N. Default tolerances: the CPU sums each row
// in double and the GPU in float with a block tree, which differ by about 1e-7
// relative. The loss is held to the same atol + rtol * |loss|.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/cross_entropy.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Rng;
using titan::Shape;
using titan::Tensor;
using titan::test::kParityAtol;
using titan::test::kParityRtol;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

std::vector<std::int32_t> random_targets(std::int64_t n, std::int64_t v,
                                         std::uint64_t seed) {
    Rng rng(seed);
    std::vector<std::int32_t> t(static_cast<std::size_t>(n));
    for (auto& x : t) x = static_cast<std::int32_t>(rng.randint(v));
    return t;
}

void expect_loss_and_grad_parity(const Shape& shape, std::uint64_t seed) {
    // Logits of scale 3, so the softmax is far from uniform.
    Tensor logits = random_tensor(shape, seed);
    for (std::int64_t i = 0; i < logits.numel(); ++i) logits.data()[i] *= 3.0f;
    const auto targets = random_targets(shape[0], shape[1], seed + 1);
    // Placeholders, as nn::CrossEntropyLoss passes: the op assigns them.
    Tensor cpu_grad = Tensor::empty({1}, Device::CPU);
    Tensor gpu_grad = Tensor::empty({1}, Device::CPU);
    const Tensor gpu_logits = logits.to(Device::CUDA);
    const float cpu_loss = ops::cross_entropy(logits, targets, &cpu_grad);
    const float gpu_loss = ops::cross_entropy(gpu_logits, targets, &gpu_grad);
    TITAN_EXPECT_NEAR(gpu_loss, cpu_loss,
                      kParityAtol + kParityRtol * std::fabs(cpu_loss));
    TITAN_EXPECT(gpu_grad.device() == Device::CUDA);
    TITAN_EXPECT(gpu_grad.shape() == shape);
    TITAN_EXPECT_PARITY(cpu_grad, gpu_grad, kParityAtol, kParityRtol);
    // Without a gradient output the loss is the same value.
    TITAN_EXPECT_EQ(ops::cross_entropy(gpu_logits, targets, nullptr), gpu_loss);
}

}  // namespace

// V = 65 (the character vocabulary of the M1 config) and V = 1000 (rows longer
// than the 256 threads of a block): a single element, a row count that is no
// multiple of 16 or 32, and 1100 x 1000 > 2^20 elements.
TITAN_TEST(CrossEntropyParity, LossAndGrad) {
    TITAN_REQUIRE_CUDA();
    const std::vector<Shape> shapes = {
        {1, 1}, {37, 65}, {512, 65}, {37, 1000}, {1100, 1000}};
    std::uint64_t seed = 1;
    for (const Shape& s : shapes) {
        expect_loss_and_grad_parity(s, seed);
        seed += 2;
    }
}

// More rows than gridDim.y allows (65535); rows go on gridDim.x.
TITAN_TEST(CrossEntropyParity, ManyRows) {
    TITAN_REQUIRE_CUDA();
    expect_loss_and_grad_parity({70000, 65}, 20);
}

// Logits of +-1000: every exponent is shifted by the row maximum, so the loss
// and the gradient stay finite on the GPU as on the CPU.
TITAN_TEST(CrossEntropyParity, StableForLargeLogits) {
    TITAN_REQUIRE_CUDA();
    const std::vector<float> big = {1000, 1001, 1002, -1000, 0, 1000};
    const Tensor logits = Tensor::from_vector(big, {2, 3});
    const std::vector<std::int32_t> targets = {0, 1};
    Tensor cpu_grad = Tensor::empty({1}, Device::CPU);
    Tensor gpu_grad = Tensor::empty({1}, Device::CPU);
    const float cpu_loss = ops::cross_entropy(logits, targets, &cpu_grad);
    const float gpu_loss =
        ops::cross_entropy(logits.to(Device::CUDA), targets, &gpu_grad);
    TITAN_EXPECT(std::isfinite(gpu_loss));
    TITAN_EXPECT_NEAR(gpu_loss, cpu_loss, 1e-3);
    TITAN_EXPECT_PARITY(cpu_grad, gpu_grad, kParityAtol, kParityRtol);
}

TITAN_TEST(CrossEntropyParity, IsRepeatable) {
    TITAN_REQUIRE_CUDA();
    const Tensor logits = random_tensor({300, 1000}, 30).to(Device::CUDA);
    const auto targets = random_targets(300, 1000, 31);
    Tensor first_grad = Tensor::empty({1}, Device::CPU);
    const float first = ops::cross_entropy(logits, targets, &first_grad);
    const std::vector<float> first_values =
        first_grad.to(Device::CPU).to_vector();
    for (int i = 0; i < 3; ++i) {
        Tensor grad = Tensor::empty({1}, Device::CPU);
        TITAN_EXPECT_EQ(ops::cross_entropy(logits, targets, &grad), first);
        TITAN_EXPECT(grad.to(Device::CPU).to_vector() == first_values);
    }
}

// Review focus 4: kernels do not validate, so an out-of-range target must be
// rejected on the host, with the CPU's message, before anything is uploaded.
TITAN_TEST(CrossEntropyParity, RejectsOutOfRangeTargetOnCuda) {
    TITAN_REQUIRE_CUDA();
    const Tensor logits = Tensor::zeros({2, 5}, Device::CUDA);
    for (std::int32_t bad : {-1, 5}) {
        const std::vector<std::int32_t> targets = {0, bad};
        Tensor grad = Tensor::empty({1}, Device::CPU);
        try {
            ops::cross_entropy(logits, targets, &grad);
            TITAN_EXPECT(false);
        } catch (const titan::Error& e) {
            TITAN_EXPECT(
                std::string(e.what()).find(
                    "ops::cross_entropy: target " + std::to_string(bad) +
                    " at index 1 is out of range [0, 5)") != std::string::npos);
        }
        // The placeholder was not replaced: the op stopped before any work.
        TITAN_EXPECT(grad.device() == Device::CPU);
    }
}
