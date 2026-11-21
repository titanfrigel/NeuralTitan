// CPU-vs-CUDA parity of RMSNorm and LayerNorm: forward outputs, saved
// statistics, dx, and the accumulated dweight and dbias.
// Default tolerances: the CPU sums in double and the GPU in float with a block
// tree, which differ by about 1e-7 relative on rows this short.

#include <cstdint>
#include <vector>

#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/norm.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Shape;
using titan::Tensor;
using titan::test::kParityAtol;
using titan::test::kParityRtol;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

constexpr float kEps = 1e-5f;

// A single element, shapes that are no multiple of 16 or 32, rows longer than
// the 256 threads of a block (300, 1025), a rank-1 row, and a tensor of at
// least 2^20 elements (1024 * 1025 = 1,049,600).
const std::vector<Shape>& parity_shapes() {
    static const std::vector<Shape> shapes = {
        {1}, {37, 129}, {3, 257}, {5, 300}, {2, 3, 7}, {7}, {1024, 1025}};
    return shapes;
}

// A weight vector near 1 so that scaled outputs stay comparable to inputs.
Tensor random_weight(std::int64_t d, std::uint64_t seed) {
    Tensor w = random_tensor({d}, seed);
    for (std::int64_t i = 0; i < d; ++i)
        w.data()[i] = 1.0f + 0.1f * w.data()[i];
    return w;
}

Tensor to_cuda(const Tensor& t) { return t.to(Device::CUDA); }

}  // namespace

TITAN_TEST(NormParity, RmsNormForward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 1);
        const Tensor w = random_weight(s.back(), 2);
        Tensor cpu_inv = Tensor::empty({1}, Device::CPU);
        Tensor gpu_inv = Tensor::empty({1}, Device::CPU);
        const Tensor cpu_y = ops::rms_norm(x, w, kEps, &cpu_inv);
        const Tensor gpu_y =
            ops::rms_norm(to_cuda(x), to_cuda(w), kEps, &gpu_inv);
        TITAN_EXPECT(cpu_inv.shape() == gpu_inv.shape());
        TITAN_EXPECT_PARITY(cpu_y, gpu_y, kParityAtol, kParityRtol);
        TITAN_EXPECT_PARITY(cpu_inv, gpu_inv, kParityAtol, kParityRtol);
    }
}

TITAN_TEST(NormParity, RmsNormForwardWithoutStatistic) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = random_tensor({37, 129}, 1);
    const Tensor w = random_weight(129, 2);
    TITAN_EXPECT_PARITY(ops::rms_norm(x, w, kEps, nullptr),
                        ops::rms_norm(to_cuda(x), to_cuda(w), kEps, nullptr),
                        kParityAtol, kParityRtol);
}

TITAN_TEST(NormParity, RmsNormBackward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const std::int64_t d = s.back();
        const Tensor x = random_tensor(s, 3);
        const Tensor w = random_weight(d, 4);
        const Tensor dy = random_tensor(s, 5);
        Tensor inv = Tensor::empty({1}, Device::CPU);
        ops::rms_norm(x, w, kEps, &inv);
        // Non-zero accumulators: backward must add to them, not overwrite.
        Tensor cpu_dw = random_tensor({d}, 6);
        Tensor gpu_dw = to_cuda(cpu_dw);
        const Tensor cpu_dx = ops::rms_norm_backward(x, w, inv, dy, cpu_dw);
        const Tensor gpu_dx = ops::rms_norm_backward(
            to_cuda(x), to_cuda(w), to_cuda(inv), to_cuda(dy), gpu_dw);
        TITAN_EXPECT_PARITY(cpu_dx, gpu_dx, kParityAtol, kParityRtol);
        TITAN_EXPECT_PARITY(cpu_dw, gpu_dw, kParityAtol, kParityRtol);
    }
}

// 2049 rows: the column kernel sums more rows than any block size, and the
// accumulators start non-zero.
TITAN_TEST(NormParity, RmsNormBackwardManyRows) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = random_tensor({2049, 33}, 7);
    const Tensor w = random_weight(33, 8);
    const Tensor dy = random_tensor({2049, 33}, 9);
    Tensor inv = Tensor::empty({1}, Device::CPU);
    ops::rms_norm(x, w, kEps, &inv);
    Tensor cpu_dw = random_tensor({33}, 10);
    Tensor gpu_dw = to_cuda(cpu_dw);
    ops::rms_norm_backward(x, w, inv, dy, cpu_dw);
    ops::rms_norm_backward(to_cuda(x), to_cuda(w), to_cuda(inv), to_cuda(dy),
                           gpu_dw);
    TITAN_EXPECT_PARITY(cpu_dw, gpu_dw, kParityAtol, kParityRtol);
}

TITAN_TEST(NormParity, LayerNormForward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 11);
        const Tensor w = random_weight(s.back(), 12);
        const Tensor b = random_tensor({s.back()}, 13);
        Tensor cpu_mean = Tensor::empty({1}, Device::CPU);
        Tensor cpu_inv = Tensor::empty({1}, Device::CPU);
        Tensor gpu_mean = Tensor::empty({1}, Device::CPU);
        Tensor gpu_inv = Tensor::empty({1}, Device::CPU);
        const Tensor cpu_y =
            ops::layer_norm(x, w, b, kEps, &cpu_mean, &cpu_inv);
        const Tensor gpu_y = ops::layer_norm(to_cuda(x), to_cuda(w), to_cuda(b),
                                             kEps, &gpu_mean, &gpu_inv);
        TITAN_EXPECT(cpu_mean.shape() == gpu_mean.shape());
        TITAN_EXPECT(cpu_inv.shape() == gpu_inv.shape());
        TITAN_EXPECT_PARITY(cpu_y, gpu_y, kParityAtol, kParityRtol);
        TITAN_EXPECT_PARITY(cpu_mean, gpu_mean, kParityAtol, kParityRtol);
        TITAN_EXPECT_PARITY(cpu_inv, gpu_inv, kParityAtol, kParityRtol);
    }
}

TITAN_TEST(NormParity, LayerNormForwardWithoutStatistics) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = random_tensor({37, 129}, 11);
    const Tensor w = random_weight(129, 12);
    const Tensor b = random_tensor({129}, 13);
    TITAN_EXPECT_PARITY(ops::layer_norm(x, w, b, kEps, nullptr, nullptr),
                        ops::layer_norm(to_cuda(x), to_cuda(w), to_cuda(b),
                                        kEps, nullptr, nullptr),
                        kParityAtol, kParityRtol);
}

TITAN_TEST(NormParity, LayerNormBackward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const std::int64_t d = s.back();
        const Tensor x = random_tensor(s, 14);
        const Tensor w = random_weight(d, 15);
        const Tensor b = random_tensor({d}, 16);
        const Tensor dy = random_tensor(s, 17);
        Tensor mean = Tensor::empty({1}, Device::CPU);
        Tensor inv = Tensor::empty({1}, Device::CPU);
        ops::layer_norm(x, w, b, kEps, &mean, &inv);
        Tensor cpu_dw = random_tensor({d}, 18);
        Tensor cpu_db = random_tensor({d}, 19);
        Tensor gpu_dw = to_cuda(cpu_dw);
        Tensor gpu_db = to_cuda(cpu_db);
        const Tensor cpu_dx =
            ops::layer_norm_backward(x, w, mean, inv, dy, cpu_dw, cpu_db);
        const Tensor gpu_dx =
            ops::layer_norm_backward(to_cuda(x), to_cuda(w), to_cuda(mean),
                                     to_cuda(inv), to_cuda(dy), gpu_dw, gpu_db);
        TITAN_EXPECT_PARITY(cpu_dx, gpu_dx, kParityAtol, kParityRtol);
        TITAN_EXPECT_PARITY(cpu_dw, gpu_dw, kParityAtol, kParityRtol);
        TITAN_EXPECT_PARITY(cpu_db, gpu_db, kParityAtol, kParityRtol);
    }
}

TITAN_TEST(NormParity, LayerNormBackwardManyRows) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = random_tensor({2049, 33}, 20);
    const Tensor w = random_weight(33, 21);
    const Tensor b = random_tensor({33}, 22);
    const Tensor dy = random_tensor({2049, 33}, 23);
    Tensor mean = Tensor::empty({1}, Device::CPU);
    Tensor inv = Tensor::empty({1}, Device::CPU);
    ops::layer_norm(x, w, b, kEps, &mean, &inv);
    Tensor cpu_dw = random_tensor({33}, 24);
    Tensor cpu_db = random_tensor({33}, 25);
    Tensor gpu_dw = to_cuda(cpu_dw);
    Tensor gpu_db = to_cuda(cpu_db);
    ops::layer_norm_backward(x, w, mean, inv, dy, cpu_dw, cpu_db);
    ops::layer_norm_backward(to_cuda(x), to_cuda(w), to_cuda(mean),
                             to_cuda(inv), to_cuda(dy), gpu_dw, gpu_db);
    TITAN_EXPECT_PARITY(cpu_dw, gpu_dw, kParityAtol, kParityRtol);
    TITAN_EXPECT_PARITY(cpu_db, gpu_db, kParityAtol, kParityRtol);
}

// More rows than gridDim.y allows (65535); rows go on gridDim.x.
TITAN_TEST(NormParity, ManyRows) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = random_tensor({70000, 7}, 26);
    const Tensor w = random_weight(7, 27);
    TITAN_EXPECT_PARITY(ops::rms_norm(x, w, kEps, nullptr),
                        ops::rms_norm(to_cuda(x), to_cuda(w), kEps, nullptr),
                        kParityAtol, kParityRtol);
}

TITAN_TEST(NormParity, IsRepeatable) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = to_cuda(random_tensor({64, 1025}, 28));
    const Tensor w = to_cuda(random_weight(1025, 29));
    const Tensor b = to_cuda(random_tensor({1025}, 30));
    const Tensor first =
        ops::layer_norm(x, w, b, kEps, nullptr, nullptr).to(Device::CPU);
    for (int i = 0; i < 3; ++i) {
        const Tensor again =
            ops::layer_norm(x, w, b, kEps, nullptr, nullptr).to(Device::CPU);
        for (std::int64_t k = 0; k < first.numel(); ++k) {
            TITAN_EXPECT_EQ(first.data()[k], again.data()[k]);
        }
    }
}
