// CPU-vs-CUDA parity of rope and rope_backward.

#include <cstdint>
#include <vector>

#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/rope.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Shape;
using titan::Tensor;
using titan::test::kParityAtol;
using titan::test::kParityRtol;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

constexpr float kTheta = 10000.0f;

// (B, H, T, hd): a single pair, shapes that are no multiple of 16 or 32, and
// one tensor of exactly 2^20 elements for each of the head dimensions 2 and 64.
const std::vector<Shape>& parity_shapes() {
    static const std::vector<Shape> shapes = {
        {1, 1, 1, 2},   {2, 3, 37, 2},   {2, 3, 37, 6},
        {1, 3, 37, 64}, {4, 8, 512, 64}, {32, 16, 1024, 2}};
    return shapes;
}

const std::int64_t kOffsets[] = {0, 7};

}  // namespace

TITAN_TEST(RopeParity, Forward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 1);
        for (const std::int64_t offset : kOffsets) {
            TITAN_EXPECT_PARITY(ops::rope(x, offset, kTheta),
                                ops::rope(x.to(Device::CUDA), offset, kTheta),
                                kParityAtol, kParityRtol);
        }
    }
}

TITAN_TEST(RopeParity, Backward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor dy = random_tensor(s, 2);
        for (const std::int64_t offset : kOffsets) {
            TITAN_EXPECT_PARITY(
                ops::rope_backward(dy, offset, kTheta),
                ops::rope_backward(dy.to(Device::CUDA), offset, kTheta),
                kParityAtol, kParityRtol);
        }
    }
}
