// CPU-vs-CUDA parity of the attention head-layout ops.

#include <cstdint>
#include <vector>

#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/layout.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Shape;
using titan::Tensor;
using titan::test::kParityAtol;
using titan::test::kParityRtol;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

// A single element, shapes that are no multiple of 16 or 32, and one tensor
// of exactly 2^20 elements. The layout ops only copy, and repeat_kv_backward
// sums its n_rep values in double like the CPU, so the default tolerances
// apply.
const std::vector<Shape>& parity_shapes() {
    static const std::vector<Shape> shapes = {
        {1, 1, 1, 1}, {3, 5, 37, 7}, {2, 3, 37, 129}, {4, 4, 512, 128}};
    return shapes;
}

const std::int64_t kReps[] = {1, 3};

}  // namespace

TITAN_TEST(LayoutParity, SwapDims12) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 1);
        TITAN_EXPECT_PARITY(ops::swap_dims_1_2(x),
                            ops::swap_dims_1_2(x.to(Device::CUDA)), kParityAtol,
                            kParityRtol);
    }
}

TITAN_TEST(LayoutParity, RepeatKv) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 2);
        for (const std::int64_t n_rep : kReps) {
            TITAN_EXPECT_PARITY(ops::repeat_kv(x, n_rep),
                                ops::repeat_kv(x.to(Device::CUDA), n_rep),
                                kParityAtol, kParityRtol);
        }
    }
}

TITAN_TEST(LayoutParity, RepeatKvBackward) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        for (const std::int64_t n_rep : kReps) {
            // dy has n_rep times the heads of the repeated input.
            const Tensor dy =
                random_tensor({s[0], s[1] * n_rep, s[2], s[3]}, 3);
            TITAN_EXPECT_PARITY(
                ops::repeat_kv_backward(dy, n_rep),
                ops::repeat_kv_backward(dy.to(Device::CUDA), n_rep),
                kParityAtol, kParityRtol);
        }
    }
}
