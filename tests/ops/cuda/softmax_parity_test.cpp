// CPU-vs-CUDA parity of softmax, causal softmax and softmax backward.
// Default tolerances: the CPU sums in double and the GPU in float with a block
// tree, which differ by about 1e-7 relative on rows this short.

#include <cstdint>
#include <vector>

#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/softmax.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Shape;
using titan::Tensor;
using titan::test::kParityAtol;
using titan::test::kParityRtol;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

// A single element, shapes that are no multiple of 16 or 32, rows longer than
// the 256 threads of a block (300, 1025, so the strided loops run more than
// once), and a tensor of at least 2^20 elements (1024 * 1025 = 1,049,600).
const std::vector<Shape>& parity_shapes() {
    static const std::vector<Shape> shapes = {
        {1}, {37, 129}, {5, 300}, {2, 3, 7}, {1024, 1025}};
    return shapes;
}

}  // namespace

TITAN_TEST(SoftmaxParity, Softmax) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 1);
        TITAN_EXPECT_PARITY(ops::softmax(x), ops::softmax(x.to(Device::CUDA)),
                            kParityAtol, kParityRtol);
    }
}

TITAN_TEST(SoftmaxParity, Causal) {
    TITAN_REQUIRE_CUDA();
    struct Case {
        Shape shape;
        std::int64_t pos_offset;
    };
    // Tq != Tk when pos_offset > 0; the last case is above 2^20 elements.
    const std::vector<Case> cases = {
        {{1, 1}, 0},      {{37, 37}, 0},       {{2, 3, 10, 15}, 5},
        {{4, 7, 300}, 5}, {{2, 512, 1025}, 0}, {{4, 3, 5, 300}, 295}};
    for (const Case& c : cases) {
        const Tensor x = random_tensor(c.shape, 2);
        TITAN_EXPECT_PARITY(
            ops::causal_softmax(x, c.pos_offset),
            ops::causal_softmax(x.to(Device::CUDA), c.pos_offset), kParityAtol,
            kParityRtol);
    }
}

TITAN_TEST(SoftmaxParity, Backward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor y = ops::softmax(random_tensor(s, 3));
        const Tensor dy = random_tensor(s, 4);
        TITAN_EXPECT_PARITY(
            ops::softmax_backward(y, dy),
            ops::softmax_backward(y.to(Device::CUDA), dy.to(Device::CUDA)),
            kParityAtol, kParityRtol);
    }
}

// More rows than gridDim.y allows (65535); rows go on gridDim.x.
TITAN_TEST(SoftmaxParity, ManyRows) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = random_tensor({70000, 7}, 5);
    TITAN_EXPECT_PARITY(ops::softmax(x), ops::softmax(x.to(Device::CUDA)),
                        kParityAtol, kParityRtol);
}

TITAN_TEST(SoftmaxParity, IsRepeatable) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = random_tensor({64, 1025}, 6).to(Device::CUDA);
    const Tensor first = ops::softmax(x).to(Device::CPU);
    for (int i = 0; i < 3; ++i) {
        const Tensor again = ops::softmax(x).to(Device::CPU);
        for (std::int64_t k = 0; k < first.numel(); ++k) {
            TITAN_EXPECT_EQ(first.data()[k], again.data()[k]);
        }
    }
}
