#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"
#include "ops/reduce.hpp"

using titan::Device;
using titan::Tensor;
namespace ops = titan::ops;

namespace {

Tensor make(std::vector<float> v, titan::Shape shape) {
    return Tensor::from_vector(v, std::move(shape));
}

}  // namespace

TITAN_TEST(Reduce, SumAndSumSquares) {
    const Tensor x = make({1, -2, 3, 4, 5, -6}, {2, 3});
    TITAN_EXPECT_EQ(ops::sum(x), 5.0);
    TITAN_EXPECT_EQ(ops::sum_squares(x), 91.0);
}

TITAN_TEST(Reduce, RowMax) {
    const Tensor x = make({1, 9, 3, -4, -5, -6}, {2, 3});
    const Tensor m = ops::row_max(x);
    TITAN_EXPECT(m.shape() == (titan::Shape{2}));
    TITAN_EXPECT(m.to_vector() == (std::vector<float>{9, -4}));

    const Tensor v = ops::row_max(make({2, 7, 5}, {3}));
    TITAN_EXPECT(v.shape() == (titan::Shape{1}));
    TITAN_EXPECT(v.to_vector() == (std::vector<float>{7}));
}

TITAN_TEST(Reduce, SumLeadingShapes) {
    std::vector<float> values(24);
    for (int i = 0; i < 24; ++i) values[i] = static_cast<float>(i);
    const Tensor s = ops::sum_leading(make(values, {2, 3, 4}));
    TITAN_EXPECT(s.shape() == (titan::Shape{4}));
    // Column c collects c, c+4, ..., c+20: 6c + 60.
    TITAN_EXPECT(s.to_vector() == (std::vector<float>{60, 66, 72, 78}));

    const Tensor r = ops::sum_leading(make({1, 2, 3}, {3}));
    TITAN_EXPECT(r.shape() == (titan::Shape{3}));
    TITAN_EXPECT(r.to_vector() == (std::vector<float>{1, 2, 3}));
}

TITAN_TEST(Reduce, SumIsDoubleAccumulated) {
    const std::int64_t n = 10'000'000;
    const std::vector<float> values(static_cast<std::size_t>(n), 0.1f);
    const Tensor x = make(values, {n});
    TITAN_EXPECT(std::abs(ops::sum(x) - 1e6) / 1e6 < 1e-3);
}

TITAN_TEST(Reduce, AllFiniteDetectsNanAndInf) {
    TITAN_EXPECT(ops::all_finite(make({1, 2, 3}, {3})));
    TITAN_EXPECT(!ops::all_finite(
        make({1, std::numeric_limits<float>::quiet_NaN(), 3}, {3})));
    TITAN_EXPECT(!ops::all_finite(
        make({1, 2, std::numeric_limits<float>::infinity()}, {3})));
    TITAN_EXPECT(!ops::all_finite(
        make({-std::numeric_limits<float>::infinity(), 2, 3}, {3})));
}

TITAN_TEST(Reduce, CudaTensorThrowsM2Message) {
    TITAN_SKIP_IF_NO_CUDA();
    const Tensor x = Tensor::zeros({2}, Device::CUDA);
    try {
        ops::sum(x);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(std::string(e.what()).find(
                         "ops::sum: CUDA implementation arrives in M2") !=
                     std::string::npos);
    }
}
