// CPU-vs-CUDA parity of the reductions.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/reduce.hpp"
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

// Whole-tensor sums come back as doubles. Both sides accumulate in double, so
// they differ only by the summation order, far below this relative bound.
void expect_close(double cpu, double gpu) {
    TITAN_EXPECT(std::fabs(cpu - gpu) <= 1e-9 + 1e-9 * std::fabs(cpu));
}

}  // namespace

TITAN_TEST(ReduceParity, Sum) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 1);
        expect_close(ops::sum(x), ops::sum(x.to(Device::CUDA)));
    }
}

TITAN_TEST(ReduceParity, SumSquares) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 2);
        expect_close(ops::sum_squares(x), ops::sum_squares(x.to(Device::CUDA)));
    }
}

TITAN_TEST(ReduceParity, SumIsRepeatable) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = random_tensor({1024, 1025}, 3).to(Device::CUDA);
    const double first = ops::sum(x);
    for (int i = 0; i < 5; ++i) TITAN_EXPECT_EQ(ops::sum(x), first);
}

TITAN_TEST(ReduceParity, RowMax) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 4);
        TITAN_EXPECT_PARITY(ops::row_max(x), ops::row_max(x.to(Device::CUDA)),
                            kParityAtol, kParityRtol);
    }
}

// More rows than gridDim.y allows (65535); rows are on gridDim.x.
TITAN_TEST(RowMaxParity, ManyRows) {
    TITAN_REQUIRE_CUDA();
    const Tensor x = random_tensor({70000, 3}, 5);
    TITAN_EXPECT_PARITY(ops::row_max(x), ops::row_max(x.to(Device::CUDA)),
                        kParityAtol, kParityRtol);
}

TITAN_TEST(ReduceParity, SumLeading) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        const Tensor x = random_tensor(s, 6);
        TITAN_EXPECT_PARITY(ops::sum_leading(x),
                            ops::sum_leading(x.to(Device::CUDA)), kParityAtol,
                            kParityRtol);
    }
    const Tensor x = random_tensor({3, 5, 7}, 7);
    TITAN_EXPECT_PARITY(ops::sum_leading(x),
                        ops::sum_leading(x.to(Device::CUDA)), kParityAtol,
                        kParityRtol);
}

TITAN_TEST(ReduceParity, AllFinite) {
    TITAN_REQUIRE_CUDA();
    for (const Shape& s : parity_shapes()) {
        TITAN_EXPECT(ops::all_finite(random_tensor(s, 8).to(Device::CUDA)));
    }
}

TITAN_TEST(AllFiniteParity, DetectsNanAndInf) {
    TITAN_REQUIRE_CUDA();
    const float bad[] = {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
    for (const Shape& s : parity_shapes()) {
        for (const float value : bad) {
            Tensor x = random_tensor(s, 9);
            std::vector<float> v = x.to_vector();
            v.back() = value;
            const Tensor c = Tensor::from_vector(v, s);
            TITAN_EXPECT(!ops::all_finite(c));
            TITAN_EXPECT(!ops::all_finite(c.to(Device::CUDA)));
        }
    }
}
