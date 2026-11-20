// CPU-vs-CUDA parity of the elementwise ops, and the device-mismatch check of
// the ops dispatch.

#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/elementwise.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Shape;
using titan::Tensor;
using titan::test::kParityAtol;
using titan::test::kParityRtol;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

// The three shape classes every parity test covers: a single element, a
// shape that is no multiple of 16 or 32, and one of at least 2^20 elements
// (1024 * 1025 = 1,049,600), which needs many blocks of the grid-stride loop.
const std::vector<Shape>& parity_shapes() {
    static const std::vector<Shape> shapes = {{1}, {37, 129}, {1024, 1025}};
    return shapes;
}

// Activations are fed N(0, 9) inputs so both tails of SiLU and GELU, where
// sigmoid and tanh saturate, are exercised as well as the middle.
Tensor wide_input(const Shape& s, std::uint64_t seed) {
    return ops::scale(random_tensor(s, seed), 3.0f);
}

std::string error_of(void (*fn)(const Tensor&, const Tensor&), const Tensor& a,
                     const Tensor& b) {
    try {
        fn(a, b);
    } catch (const titan::Error& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TITAN_TEST(OpsDispatch, MixedDevicesThrow) {
    TITAN_REQUIRE_CUDA();
    const Tensor cpu = Tensor::zeros({2, 3}, Device::CPU);
    const Tensor gpu = Tensor::zeros({2, 3}, Device::CUDA);
    const auto add = [](const Tensor& a, const Tensor& b) { ops::add(a, b); };

    // The devices are named in argument order.
    const std::string cpu_first = error_of(add, cpu, gpu);
    TITAN_EXPECT(cpu_first.find("ops::add: tensors are on different devices "
                                "(cpu vs cuda)") != std::string::npos);
    const std::string cuda_first = error_of(add, gpu, cpu);
    TITAN_EXPECT(cuda_first.find("ops::add: tensors are on different devices "
                                 "(cuda vs cpu)") != std::string::npos);

    // In-place and backward ops reject a mix of devices too.
    Tensor target = Tensor::zeros({2, 3}, Device::CUDA);
    TITAN_EXPECT_THROWS(ops::add_(target, cpu), titan::Error);
    TITAN_EXPECT_THROWS(ops::silu_backward(cpu, gpu), titan::Error);
}

TITAN_TEST(ElementwiseParity, Add) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor a = random_tensor(s, 1);
        const Tensor b = random_tensor(s, 2);
        TITAN_EXPECT_PARITY(ops::add(a, b),
                            ops::add(a.to(Device::CUDA), b.to(Device::CUDA)),
                            kParityAtol, kParityRtol);
    }
}

TITAN_TEST(ElementwiseParity, AddInPlace) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        Tensor c = random_tensor(s, 3);
        Tensor g = c.to(Device::CUDA);
        const Tensor b = random_tensor(s, 4);
        ops::add_(c, b);
        ops::add_(g, b.to(Device::CUDA));
        TITAN_EXPECT_PARITY(c, g, kParityAtol, kParityRtol);
    }
}

TITAN_TEST(ElementwiseParity, AddRowwiseInPlace) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        Tensor c = random_tensor(s, 5);
        Tensor g = c.to(Device::CUDA);
        const Tensor b = random_tensor({s.back()}, 6);
        ops::add_rowwise_(c, b);
        ops::add_rowwise_(g, b.to(Device::CUDA));
        TITAN_EXPECT_PARITY(c, g, kParityAtol, kParityRtol);
    }
}

TITAN_TEST(ElementwiseParity, Mul) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor a = random_tensor(s, 7);
        const Tensor b = random_tensor(s, 8);
        TITAN_EXPECT_PARITY(ops::mul(a, b),
                            ops::mul(a.to(Device::CUDA), b.to(Device::CUDA)),
                            kParityAtol, kParityRtol);
    }
}

TITAN_TEST(ElementwiseParity, Scale) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 9);
        TITAN_EXPECT_PARITY(ops::scale(x, -0.37f),
                            ops::scale(x.to(Device::CUDA), -0.37f), kParityAtol,
                            kParityRtol);
    }
}

TITAN_TEST(ElementwiseParity, ScaleInPlace) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        Tensor c = random_tensor(s, 10);
        Tensor g = c.to(Device::CUDA);
        ops::scale_(c, 1.7f);
        ops::scale_(g, 1.7f);
        TITAN_EXPECT_PARITY(c, g, kParityAtol, kParityRtol);
    }
}

TITAN_TEST(ElementwiseParity, Silu) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = wide_input(s, 11);
        TITAN_EXPECT_PARITY(ops::silu(x), ops::silu(x.to(Device::CUDA)),
                            kParityAtol, kParityRtol);
    }
}

TITAN_TEST(ElementwiseParity, SiluBackward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = wide_input(s, 12);
        const Tensor dy = random_tensor(s, 13);
        TITAN_EXPECT_PARITY(
            ops::silu_backward(x, dy),
            ops::silu_backward(x.to(Device::CUDA), dy.to(Device::CUDA)),
            kParityAtol, kParityRtol);
    }
}

TITAN_TEST(ElementwiseParity, Gelu) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = wide_input(s, 14);
        TITAN_EXPECT_PARITY(ops::gelu(x), ops::gelu(x.to(Device::CUDA)),
                            kParityAtol, kParityRtol);
    }
}

TITAN_TEST(ElementwiseParity, GeluBackward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = wide_input(s, 15);
        const Tensor dy = random_tensor(s, 16);
        TITAN_EXPECT_PARITY(
            ops::gelu_backward(x, dy),
            ops::gelu_backward(x.to(Device::CUDA), dy.to(Device::CUDA)),
            kParityAtol, kParityRtol);
    }
}
