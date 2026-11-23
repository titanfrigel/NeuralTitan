#pragma once

#include <cstdint>
#include <string>

#include "core/tensor.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"

namespace titan::test {

// Default CPU-vs-CUDA tolerances. An op that needs looser ones states them in
// a comment with the reason.
inline constexpr double kParityAtol = 1e-5;
inline constexpr double kParityRtol = 1e-4;

// A CPU tensor of the given shape drawn from N(0, 1) by an Rng seeded with
// `seed`, so a test can rebuild the same input from the seed alone.
Tensor random_tensor(Shape s, std::uint64_t seed);

struct ParityResult {
    bool ok;
    // Empty when ok. Otherwise the reason: a shape or device mismatch, or the
    // worst element's flat index with the CPU and CUDA values.
    std::string detail;
};

// Compares a CPU result with a CUDA result of the same op. `cpu` must be on
// the CPU and `cuda` on the GPU, with equal shapes. Every element must satisfy
// |c - g| <= atol + rtol * |c|, evaluated in double; a NaN on either side
// fails. The worst element is the one exceeding its bound by the most.
ParityResult check_parity(const Tensor& cpu, const Tensor& cuda, double atol,
                          double rtol);

}  // namespace titan::test

// Skips the running test when the machine has no CUDA device. Use it as a
// statement at the top of every CUDA test: TITAN_REQUIRE_CUDA();
#define TITAN_REQUIRE_CUDA() TITAN_SKIP_IF_NO_CUDA()

// Records one failure naming the worst element unless the CPU tensor `c` and
// the CUDA tensor `g` agree within `atol + rtol * |c|` everywhere.
#define TITAN_EXPECT_PARITY(c, g, atol, rtol)                      \
    do {                                                           \
        const ::titan::test::ParityResult titan_parity_result =    \
            ::titan::test::check_parity((c), (g), (atol), (rtol)); \
        if (!titan_parity_result.ok) {                             \
            ::titan::test::report_failure(                         \
                __FILE__, __LINE__,                                \
                std::string("parity " #c " vs " #g ": ") +         \
                    titan_parity_result.detail);                   \
        }                                                          \
    } while (false)
