// CPU-vs-CUDA parity of the AdamW update, and the device-mismatch check of
// its dispatch.

#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/adamw.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Shape;
using titan::Tensor;
using titan::test::kParityAtol;
using titan::test::kParityRtol;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

// A single element, a shape that is no multiple of 16 or 32, and one of at
// least 2^20 elements (1024 * 1025 = 1,049,600).
const std::vector<Shape>& parity_shapes() {
    static const std::vector<Shape> shapes = {{1}, {37, 129}, {1024, 1025}};
    return shapes;
}

// Ten updates from the same state on both devices, with a fresh gradient each
// step and weight decay on, so the moments, the bias correction (steps 1..10,
// where it is far from 1) and the decoupled decay are all exercised.
void expect_ten_steps_match(const Shape& shape) {
    Tensor p_cpu = random_tensor(shape, 1);
    Tensor m_cpu = Tensor::zeros(shape, Device::CPU);
    Tensor v_cpu = Tensor::zeros(shape, Device::CPU);
    Tensor p_gpu = p_cpu.to(Device::CUDA);
    Tensor m_gpu = m_cpu.to(Device::CUDA);
    Tensor v_gpu = v_cpu.to(Device::CUDA);
    for (std::int64_t step = 1; step <= 10; ++step) {
        const Tensor g_cpu =
            random_tensor(shape, 100 + static_cast<std::uint64_t>(step));
        const Tensor g_gpu = g_cpu.to(Device::CUDA);
        const ops::AdamWHyper h{1e-2f, 0.9f, 0.999f, 1e-8f, 0.1f, step};
        ops::adamw_update(p_cpu, g_cpu, m_cpu, v_cpu, h);
        ops::adamw_update(p_gpu, g_gpu, m_gpu, v_gpu, h);
    }
    TITAN_EXPECT_PARITY(p_cpu, p_gpu, kParityAtol, kParityRtol);
    TITAN_EXPECT_PARITY(m_cpu, m_gpu, kParityAtol, kParityRtol);
    TITAN_EXPECT_PARITY(v_cpu, v_gpu, kParityAtol, kParityRtol);
}

}  // namespace

TITAN_TEST(AdamWParity, TenSteps) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& shape : parity_shapes()) expect_ten_steps_match(shape);
}

TITAN_TEST(AdamWParity, MixedDevicesThrow) {
    TITAN_REQUIRE_CUDA();
    Tensor p = Tensor::zeros({2}, Device::CUDA);
    const Tensor g = Tensor::zeros({2}, Device::CPU);
    Tensor m = Tensor::zeros({2}, Device::CUDA);
    Tensor v = Tensor::zeros({2}, Device::CUDA);
    const ops::AdamWHyper h{1e-2f, 0.9f, 0.999f, 1e-8f, 0.1f, 1};
    try {
        ops::adamw_update(p, g, m, v, h);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(std::string(e.what()).find(
                         "ops::adamw_update: tensors are on different devices "
                         "(cuda vs cpu)") != std::string::npos);
    }
}
