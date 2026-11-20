#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"
#include "ops/layout.hpp"
#include "support/gradcheck.hpp"

using titan::Device;
using titan::Rng;
using titan::Tensor;
using titan::test::check_gradient;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

// 0, 1, 2, ... in row-major order, so every element names its own index.
Tensor iota(titan::Shape shape) {
    Tensor t = Tensor::empty(shape, Device::CPU);
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        t.data()[i] = static_cast<float>(i);
    }
    return t;
}

void expect_cuda_message(const std::string& message, const std::string& op) {
    TITAN_EXPECT(
        message.find("ops::" + op + ": CUDA implementation arrives in M2") !=
        std::string::npos);
}

}  // namespace

TITAN_TEST(Layout, SwapDims12Values) {
    // x is (A=2, B=3, C=4, D=5); x[a, b, c, d] = ((a·3 + b)·4 + c)·5 + d.
    const Tensor x = iota({2, 3, 4, 5});
    const Tensor y = ops::swap_dims_1_2(x);
    TITAN_EXPECT(y.shape() == titan::Shape({2, 4, 3, 5}));
    const std::vector<float> v = y.to_vector();
    for (int a = 0; a < 2; ++a) {
        for (int c = 0; c < 4; ++c) {
            for (int b = 0; b < 3; ++b) {
                for (int d = 0; d < 5; ++d) {
                    const float expected =
                        static_cast<float>(((a * 3 + b) * 4 + c) * 5 + d);
                    TITAN_EXPECT_EQ(v[((a * 4 + c) * 3 + b) * 5 + d], expected);
                }
            }
        }
    }
}

TITAN_TEST(Layout, SwapDims12IsItsOwnInverse) {
    Rng rng(1);
    const Tensor x = random_tensor({2, 3, 4, 5}, rng);
    const Tensor back = ops::swap_dims_1_2(ops::swap_dims_1_2(x));
    TITAN_EXPECT(back.shape() == x.shape());
    TITAN_EXPECT(back.to_vector() == x.to_vector());
}

TITAN_TEST(Layout, RepeatKvValues) {
    // x is (B=2, Hkv=2, T=1, hd=2): head g of batch b holds (4b + 2g, +1).
    const Tensor x = iota({2, 2, 1, 2});
    const Tensor y = ops::repeat_kv(x, 3);
    TITAN_EXPECT(y.shape() == titan::Shape({2, 6, 1, 2}));
    const std::vector<float> v = y.to_vector();
    for (int b = 0; b < 2; ++b) {
        for (int h = 0; h < 6; ++h) {
            const int g = h / 3;
            for (int d = 0; d < 2; ++d) {
                TITAN_EXPECT_EQ(v[(b * 6 + h) * 2 + d],
                                static_cast<float>((b * 2 + g) * 2 + d));
            }
        }
    }
}

TITAN_TEST(Layout, RepeatKvOneIsIdentity) {
    Rng rng(2);
    const Tensor x = random_tensor({2, 3, 4, 2}, rng);
    TITAN_EXPECT(ops::repeat_kv(x, 1).to_vector() == x.to_vector());
    TITAN_EXPECT(ops::repeat_kv_backward(x, 1).to_vector() == x.to_vector());
}

TITAN_TEST(Layout, RepeatKvBackwardSumsGroups) {
    // dy is (B=1, H=4, T=1, hd=1) = [0, 1, 2, 3]; groups of 2 sum to 1 and 5.
    const Tensor dx = ops::repeat_kv_backward(iota({1, 4, 1, 1}), 2);
    TITAN_EXPECT(dx.shape() == titan::Shape({1, 2, 1, 1}));
    TITAN_EXPECT(dx.to_vector() == std::vector<float>({1.0f, 5.0f}));
}

TITAN_TEST(Layout, RepeatKvBackwardOfRepeatScalesByNRep) {
    // Summing n_rep copies of each head gives n_rep times the head.
    Rng rng(3);
    const Tensor x = random_tensor({2, 2, 3, 4}, rng);
    const std::vector<float> back =
        ops::repeat_kv_backward(ops::repeat_kv(x, 3), 3).to_vector();
    const std::vector<float> v = x.to_vector();
    for (std::size_t i = 0; i < v.size(); ++i) {
        TITAN_EXPECT_NEAR(back[i], 3.0 * v[i], 1e-6);
    }
}

TITAN_TEST(Layout, RepeatKvGradCheck) {
    Rng rng(4);
    Tensor x = random_tensor({2, 2, 3, 2}, rng);
    const Tensor r = random_tensor({2, 4, 3, 2}, rng);
    const Tensor grad = ops::repeat_kv_backward(r, 2);
    TITAN_EXPECT_GRAD_OK(
        check_gradient([&] { return ops::repeat_kv(x, 2); }, x, grad, r));
}

TITAN_TEST(Layout, SwapDims12GradCheck) {
    Rng rng(5);
    Tensor x = random_tensor({2, 3, 2, 2}, rng);
    const Tensor r = random_tensor({2, 2, 3, 2}, rng);
    const Tensor grad = ops::swap_dims_1_2(r);
    TITAN_EXPECT_GRAD_OK(
        check_gradient([&] { return ops::swap_dims_1_2(x); }, x, grad, r));
}

TITAN_TEST(Layout, InvalidArgumentsThrow) {
    const Tensor rank3 = Tensor::zeros({2, 2, 2}, Device::CPU);
    const Tensor x = Tensor::zeros({1, 4, 2, 2}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::swap_dims_1_2(rank3), titan::Error);
    TITAN_EXPECT_THROWS(ops::repeat_kv(rank3, 2), titan::Error);
    TITAN_EXPECT_THROWS(ops::repeat_kv(x, 0), titan::Error);
    TITAN_EXPECT_THROWS(ops::repeat_kv_backward(rank3, 2), titan::Error);
    TITAN_EXPECT_THROWS(ops::repeat_kv_backward(x, 0), titan::Error);
    TITAN_EXPECT_THROWS(ops::repeat_kv_backward(x, 3), titan::Error);
}

TITAN_TEST(Layout, CudaTensorThrowsM2Message) {
    TITAN_SKIP_IF_NO_CUDA();
    const Tensor x = Tensor::zeros({1, 2, 2, 2}, Device::CUDA);
    try {
        ops::swap_dims_1_2(x);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        expect_cuda_message(e.what(), "swap_dims_1_2");
    }
    try {
        ops::repeat_kv(x, 2);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        expect_cuda_message(e.what(), "repeat_kv");
    }
    try {
        ops::repeat_kv_backward(x, 2);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        expect_cuda_message(e.what(), "repeat_kv_backward");
    }
}
