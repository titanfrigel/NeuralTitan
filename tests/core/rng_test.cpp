#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"

using titan::Rng;

TITAN_TEST(Rng, MatchesStandardEngine) {
    Rng rng(5489);
    std::mt19937_64 reference(5489);
    TITAN_EXPECT(rng.next_u64() == reference());
}

TITAN_TEST(Rng, SameSeedSameSequence) {
    Rng a(42);
    Rng b(42);
    bool same = true;
    for (int i = 0; i < 1000; ++i) {
        same = same && (a.uniform() == b.uniform());
    }
    TITAN_EXPECT(same);
}

TITAN_TEST(Rng, UniformInRange) {
    Rng rng(1);
    const int n = 100000;
    double sum = 0.0;
    bool in_range = true;
    for (int i = 0; i < n; ++i) {
        float u = rng.uniform();
        in_range = in_range && u >= 0.0f && u < 1.0f;
        sum += u;
    }
    TITAN_EXPECT(in_range);
    TITAN_EXPECT_NEAR(sum / n, 0.5, 0.01);
}

TITAN_TEST(Rng, NormalMoments) {
    Rng rng(2);
    const int n = 100000;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        double z = rng.normal();
        sum += z;
        sum_sq += z * z;
    }
    double mean = sum / n;
    double variance = sum_sq / n - mean * mean;
    TITAN_EXPECT_NEAR(mean, 0.0, 0.02);
    TITAN_EXPECT_NEAR(variance, 1.0, 0.03);
}

TITAN_TEST(Rng, RandintBounds) {
    Rng rng(3);
    std::vector<int> counts(8, 0);
    for (int i = 0; i < 70000; ++i) {
        std::int64_t v = rng.randint(7);
        TITAN_EXPECT(v >= 0 && v < 7);
        if (v >= 0 && v < 8) {
            ++counts[static_cast<std::size_t>(v)];
        }
    }
    TITAN_EXPECT_EQ(counts[7], 0);
    for (std::size_t k = 0; k < 7; ++k) {
        TITAN_EXPECT_NEAR(counts[k], 10000, 1000);
    }
    TITAN_EXPECT_THROWS(rng.randint(0), titan::Error);
}

TITAN_TEST(Rng, FillNormalIsSeeded) {
    auto t1 = titan::Tensor::zeros({4, 5}, titan::Device::CPU);
    auto t2 = titan::Tensor::zeros({4, 5}, titan::Device::CPU);
    Rng a(9);
    Rng b(9);
    a.fill_normal(t1, 1.0f, 2.0f);
    b.fill_normal(t2, 1.0f, 2.0f);
    TITAN_EXPECT(t1.to_vector() == t2.to_vector());
    TITAN_EXPECT(t1.to_vector()[0] != 0.0f);
}
