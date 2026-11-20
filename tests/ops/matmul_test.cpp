#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/parallel.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/cpu/matmul.hpp"
#include "ops/matmul.hpp"
#include "support/gradcheck.hpp"

using titan::Device;
using titan::Rng;
using titan::Shape;
using titan::Tensor;
using titan::test::check_gradient;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

Tensor make(std::vector<float> v, Shape shape) {
    return Tensor::from_vector(v, std::move(shape));
}

// Swaps the last two dimensions by copying.
Tensor transposed(const Tensor& x) {
    const std::int64_t r = x.dim(-2), c = x.dim(-1);
    const std::int64_t batch = x.numel() / (r * c);
    Shape s = x.shape();
    std::swap(s[s.size() - 1], s[s.size() - 2]);
    Tensor out = Tensor::empty(s, Device::CPU);
    for (std::int64_t t = 0; t < batch; ++t) {
        for (std::int64_t i = 0; i < r; ++i) {
            for (std::int64_t j = 0; j < c; ++j) {
                out.data()[t * r * c + j * r + i] =
                    x.data()[t * r * c + i * c + j];
            }
        }
    }
    return out;
}

// Largest |x - y| / (1 + |y|) over all elements; y is the reference.
double max_relative_error(const Tensor& x, const Tensor& y) {
    double worst = 0.0;
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        const double ref = y.data()[i];
        worst = std::max(worst,
                         std::abs(x.data()[i] - ref) / (1.0 + std::abs(ref)));
    }
    return worst;
}

Tensor naive(const Tensor& a, const Tensor& b) {
    Shape s(a.shape().begin(), a.shape().end() - 2);
    s.push_back(a.dim(-2));
    s.push_back(b.dim(-1));
    Tensor out = Tensor::zeros(s, Device::CPU);
    ops::cpu::matmul_naive(out, a, b, false, false);
    return out;
}

// Restores the thread count when a test ends.
struct ThreadGuard {
    int previous = titan::num_threads();
    ~ThreadGuard() { titan::set_num_threads(previous); }
};

}  // namespace

TITAN_TEST(Matmul, Small2D) {
    const Tensor a = make({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor b = make({7, 8, 9, 10, 11, 12}, {3, 2});
    const Tensor c = ops::matmul(a, b);
    TITAN_EXPECT(c.shape() == (Shape{2, 2}));
    TITAN_EXPECT(c.to_vector() == (std::vector<float>{58, 64, 139, 154}));
}

TITAN_TEST(Matmul, Transposes) {
    Rng rng(1);
    const Tensor a = random_tensor({5, 7}, rng);
    const Tensor b = random_tensor({7, 4}, rng);
    const Tensor expected = ops::matmul(a, b);
    const Tensor at = transposed(a);
    const Tensor bt = transposed(b);
    TITAN_EXPECT(ops::matmul(at, b, true, false).to_vector() ==
                 expected.to_vector());
    TITAN_EXPECT(ops::matmul(a, bt, false, true).to_vector() ==
                 expected.to_vector());
    TITAN_EXPECT(ops::matmul(at, bt, true, true).to_vector() ==
                 expected.to_vector());
}

TITAN_TEST(Matmul, Batched3DAnd4D) {
    Rng rng(2);
    const Tensor a3 = random_tensor({3, 4, 5}, rng);
    const Tensor b3 = random_tensor({3, 5, 6}, rng);
    const Tensor c3 = ops::matmul(a3, b3);
    TITAN_EXPECT(c3.shape() == (Shape{3, 4, 6}));
    for (std::int64_t t = 0; t < 3; ++t) {
        Tensor a2 = Tensor::empty({4, 5}, Device::CPU);
        Tensor b2 = Tensor::empty({5, 6}, Device::CPU);
        std::copy_n(a3.data() + t * 20, 20, a2.data());
        std::copy_n(b3.data() + t * 30, 30, b2.data());
        const Tensor c2 = ops::matmul(a2, b2);
        TITAN_EXPECT(std::equal(c2.data(), c2.data() + 24, c3.data() + t * 24));
    }

    const Tensor a4 = random_tensor({2, 3, 4, 5}, rng);
    const Tensor b4 = random_tensor({2, 3, 5, 6}, rng);
    const Tensor c4 = ops::matmul(a4, b4);
    TITAN_EXPECT(c4.shape() == (Shape{2, 3, 4, 6}));
    for (std::int64_t t = 0; t < 6; ++t) {
        Tensor a2 = Tensor::empty({4, 5}, Device::CPU);
        Tensor b2 = Tensor::empty({5, 6}, Device::CPU);
        std::copy_n(a4.data() + t * 20, 20, a2.data());
        std::copy_n(b4.data() + t * 30, 30, b2.data());
        const Tensor c2 = ops::matmul(a2, b2);
        TITAN_EXPECT(std::equal(c2.data(), c2.data() + 24, c4.data() + t * 24));
    }
}

TITAN_TEST(Matmul, MatchesNaiveOnAwkwardShapes) {
    Rng rng(3);
    const Tensor a = random_tensor({37, 53}, rng);
    const Tensor b = random_tensor({53, 29}, rng);
    TITAN_EXPECT(max_relative_error(ops::matmul(a, b), naive(a, b)) < 1e-4);

    const Tensor a3 = random_tensor({3, 17, 65}, rng);
    const Tensor b3 = random_tensor({3, 65, 31}, rng);
    TITAN_EXPECT(max_relative_error(ops::matmul(a3, b3), naive(a3, b3)) < 1e-4);

    // A K longer than one block, with both transposes.
    const Tensor a4 = random_tensor({200, 9}, rng);
    const Tensor b4 = random_tensor({13, 200}, rng);
    const Tensor want = naive(transposed(a4), transposed(b4));
    TITAN_EXPECT(max_relative_error(ops::matmul(a4, b4, true, true), want) <
                 1e-4);
}

TITAN_TEST(Matmul, DeterministicAcrossThreadCounts) {
    const ThreadGuard guard;
    Rng rng(4);
    const Tensor a = random_tensor({2, 61, 150}, rng);
    const Tensor b = random_tensor({2, 150, 47}, rng);
    titan::set_num_threads(1);
    const Tensor one = ops::matmul(a, b);
    titan::set_num_threads(8);
    const Tensor eight = ops::matmul(a, b);
    TITAN_EXPECT(
        std::equal(one.data(), one.data() + one.numel(), eight.data()));
}

TITAN_TEST(Matmul, AccAccumulates) {
    const Tensor a = make({1, 2, 3, 4}, {2, 2});
    const Tensor b = make({1, 0, 0, 1}, {2, 2});
    Tensor out = make({10, 10, 10, 10}, {2, 2});
    ops::matmul_acc(out, a, b, false, false);
    ops::matmul_acc(out, a, b, false, false);
    TITAN_EXPECT(out.to_vector() == (std::vector<float>{12, 14, 16, 18}));
}

TITAN_TEST(Matmul, ShapeMismatchThrows) {
    const Tensor a = Tensor::zeros({2, 3}, Device::CPU);
    const Tensor b = Tensor::zeros({4, 5}, Device::CPU);
    try {
        ops::matmul(a, b, false, true);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        const std::string msg = e.what();
        TITAN_EXPECT(msg.find("[2, 3]") != std::string::npos);
        TITAN_EXPECT(msg.find("[4, 5]") != std::string::npos);
        TITAN_EXPECT(msg.find("transpose_a=false") != std::string::npos);
        TITAN_EXPECT(msg.find("transpose_b=true") != std::string::npos);
    }
    const Tensor batched = Tensor::zeros({2, 3, 4}, Device::CPU);
    const Tensor other = Tensor::zeros({3, 4, 5}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::matmul(batched, other), titan::Error);
    TITAN_EXPECT_THROWS(ops::matmul(a, batched), titan::Error);
    Tensor wrong_out = Tensor::zeros({2, 2}, Device::CPU);
    const Tensor c = Tensor::zeros({3, 4}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::matmul_acc(wrong_out, a, c, false, false),
                        titan::Error);
}

TITAN_TEST(Matmul, GradCheck) {
    Rng rng(5);
    Tensor a = random_tensor({4, 6}, rng);
    Tensor b = random_tensor({6, 5}, rng);
    const Tensor r = random_tensor({4, 5}, rng);
    // L = sum(C ⊙ r), so dC = r.
    const Tensor da = ops::matmul(r, b, false, true);
    const Tensor db = ops::matmul(a, r, true, false);
    TITAN_EXPECT_GRAD_OK(
        check_gradient([&] { return ops::matmul(a, b); }, a, da, r));
    TITAN_EXPECT_GRAD_OK(
        check_gradient([&] { return ops::matmul(a, b); }, b, db, r));
}
