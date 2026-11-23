#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/module.hpp"
#include "optim/adamw.hpp"

using titan::Tensor;
using titan::nn::Parameter;
namespace optim = titan::optim;

namespace {

// Two parameters (rank 2 and rank 1) with fixed values.
struct Params {
    Parameter w = titan::nn::make_parameter("w", {2, 3});
    Parameter b = titan::nn::make_parameter("b", {3});

    Params() {
        float* pw = w.value.data();
        for (int i = 0; i < 6; ++i) pw[i] = 0.1f * static_cast<float>(i + 1);
        float* pb = b.value.data();
        for (int i = 0; i < 3; ++i) pb[i] = -0.2f * static_cast<float>(i + 1);
    }

    // A deterministic gradient that depends on the step index.
    void set_grads(int step) {
        float* gw = w.grad.data();
        for (int i = 0; i < 6; ++i) {
            gw[i] = 0.3f * static_cast<float>((i + step) % 4) - 0.4f;
        }
        float* gb = b.grad.data();
        for (int i = 0; i < 3; ++i) {
            gb[i] = 0.5f - 0.25f * static_cast<float>((i * step) % 3);
        }
    }

    void set_values(const std::vector<float>& all) {
        float* pw = w.value.data();
        for (int i = 0; i < 6; ++i) pw[i] = all[static_cast<std::size_t>(i)];
        float* pb = b.value.data();
        for (int i = 0; i < 3; ++i) {
            pb[i] = all[static_cast<std::size_t>(6 + i)];
        }
    }

    std::vector<float> values() const {
        std::vector<float> all = w.value.to_vector();
        const std::vector<float> bv = b.value.to_vector();
        all.insert(all.end(), bv.begin(), bv.end());
        return all;
    }
};

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) !=
            std::bit_cast<std::uint32_t>(b[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace

TITAN_TEST(AdamW, StateRoundTripContinuesIdentically) {
    Params a;
    optim::AdamW opt_a({&a.w, &a.b}, optim::AdamWConfig{});
    for (int s = 0; s < 3; ++s) {
        a.set_grads(s);
        opt_a.step(0.01f);
    }
    const optim::AdamWState saved = opt_a.state();
    const std::vector<float> at_save = a.values();
    TITAN_EXPECT_EQ(saved.steps, std::int64_t{3});
    for (int s = 3; s < 5; ++s) {
        a.set_grads(s);
        opt_a.step(0.01f);
    }
    // The saved state is a copy: the two further steps did not change it.
    TITAN_EXPECT_EQ(saved.steps, std::int64_t{3});

    Params b;
    b.set_values(at_save);
    optim::AdamW opt_b({&b.w, &b.b}, optim::AdamWConfig{});
    opt_b.load_state(saved);
    TITAN_EXPECT_EQ(opt_b.steps(), std::int64_t{3});
    for (int s = 3; s < 5; ++s) {
        b.set_grads(s);
        opt_b.step(0.01f);
    }
    TITAN_EXPECT(bits_equal(a.values(), b.values()));
    TITAN_EXPECT_EQ(opt_b.steps(), std::int64_t{5});
}

TITAN_TEST(AdamW, LoadStateRejectsShapeMismatch) {
    Params p;
    optim::AdamW opt({&p.w, &p.b}, optim::AdamWConfig{});

    optim::AdamWState bad_shape = opt.state();
    bad_shape.v[1] = Tensor::zeros({4}, titan::Device::CPU);
    std::string message;
    try {
        opt.load_state(bad_shape);
    } catch (const titan::Error& e) {
        message = e.what();
    }
    TITAN_EXPECT(message.find("parameter b") != std::string::npos);

    optim::AdamWState bad_count = opt.state();
    bad_count.m.pop_back();
    TITAN_EXPECT_THROWS(opt.load_state(bad_count), titan::Error);
}
