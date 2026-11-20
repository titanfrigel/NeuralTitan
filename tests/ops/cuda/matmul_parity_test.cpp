// CPU-vs-CUDA parity of the matmul for each GEMM backend (cuBLAS, naive,
// tiled), and the GEMM backend switch.

#include <cstdint>
#include <vector>

#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/matmul.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Shape;
using titan::Tensor;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

// Looser than the default parity tolerances: each output is a float sum of K
// products, and the CPU and the GPU add them in different orders (the CPU in
// blocks of 64 with a double total, cuBLAS in its own order, the naive and
// tiled kernels in one float accumulator), so the rounding grows with K.
constexpr double kMatmulAtol = 1e-4;
constexpr double kMatmulRtol = 1e-4;

// Selects a GEMM backend for the lifetime of the guard and restores the
// previous one afterwards, so no test leaks its choice into the next.
class GemmBackendGuard {
public:
    explicit GemmBackendGuard(ops::GemmBackend backend)
        : previous_(ops::gemm_backend()) {
        ops::set_gemm_backend(backend);
    }
    ~GemmBackendGuard() { ops::set_gemm_backend(previous_); }
    GemmBackendGuard(const GemmBackendGuard&) = delete;
    GemmBackendGuard& operator=(const GemmBackendGuard&) = delete;

private:
    ops::GemmBackend previous_;
};

// The stored shape of an operand whose op() is (rows, cols).
Shape stored(Shape batch, std::int64_t rows, std::int64_t cols,
             bool transpose) {
    batch.push_back(transpose ? cols : rows);
    batch.push_back(transpose ? rows : cols);
    return batch;
}

// Multiplies op(a) (..., M, K) by op(b) (..., K, N) on both devices and
// compares the results.
void expect_matmul_parity(const Shape& batch, std::int64_t m, std::int64_t k,
                          std::int64_t n, bool ta, bool tb,
                          std::uint64_t seed) {
    const Tensor a = random_tensor(stored(batch, m, k, ta), seed);
    const Tensor b = random_tensor(stored(batch, k, n, tb), seed + 1);
    const Tensor cpu = ops::matmul(a, b, ta, tb);
    const Tensor gpu =
        ops::matmul(a.to(Device::CUDA), b.to(Device::CUDA), ta, tb);
    TITAN_EXPECT_PARITY(cpu, gpu, kMatmulAtol, kMatmulRtol);
}

// The cases every backend must match the CPU on:
// - all four transpose combinations, on a rank-2 shape that is no multiple of
//   16 and on the batched rank-4 shape (2, 3, 37, 129) x (2, 3, 129, 65);
// - a single element, (1, 1) x (1, 1);
// - a 512 x 512 x 512 product;
// - a result of more than 2^20 elements whose 65,537 row tiles exceed the
//   tiled kernel's 65,535-block grid, so its tile loop runs more than once.
void run_parity_cases(ops::GemmBackend backend) {
    const GemmBackendGuard guard(backend);
    std::uint64_t seed = 100;
    for (const bool ta : {false, true}) {
        for (const bool tb : {false, true}) {
            expect_matmul_parity({}, 37, 129, 65, ta, tb, seed);
            expect_matmul_parity({2, 3}, 37, 129, 65, ta, tb, seed + 2);
            seed += 4;
        }
    }
    expect_matmul_parity({}, 1, 1, 1, false, false, 200);
    expect_matmul_parity({}, 512, 512, 512, false, false, 300);
    expect_matmul_parity({}, 1048583, 3, 2, false, false, 400);
}

}  // namespace

TITAN_TEST(Matmul, DefaultBackendIsCublas) {
    TITAN_EXPECT(ops::gemm_backend() == ops::GemmBackend::cublas);
}

TITAN_TEST(Matmul, SetGemmBackendRoundTrips) {
    for (const ops::GemmBackend backend :
         {ops::GemmBackend::naive, ops::GemmBackend::tiled,
          ops::GemmBackend::cublas}) {
        const GemmBackendGuard guard(backend);
        TITAN_EXPECT(ops::gemm_backend() == backend);
    }
    TITAN_EXPECT(ops::gemm_backend() == ops::GemmBackend::cublas);
}

TITAN_TEST(Matmul, CpuIgnoresGemmBackend) {
    const Tensor a = random_tensor({5, 7}, 1);
    const Tensor b = random_tensor({7, 3}, 2);
    const std::vector<float> expected = ops::matmul(a, b).to_vector();
    const GemmBackendGuard guard(ops::GemmBackend::naive);
    TITAN_EXPECT(ops::matmul(a, b).to_vector() == expected);
}

TITAN_TEST(MatmulParity, Cublas) {
    TITAN_REQUIRE_CUDA();
    run_parity_cases(ops::GemmBackend::cublas);
}

TITAN_TEST(MatmulParity, Naive) {
    TITAN_REQUIRE_CUDA();
    run_parity_cases(ops::GemmBackend::naive);
}

TITAN_TEST(MatmulParity, Tiled) {
    TITAN_REQUIRE_CUDA();
    run_parity_cases(ops::GemmBackend::tiled);
}

// matmul_acc adds to a non-zero out on every backend and transpose case.
TITAN_TEST(MatmulParity, AccAddsToOut) {
    TITAN_REQUIRE_CUDA();
    for (const ops::GemmBackend backend :
         {ops::GemmBackend::cublas, ops::GemmBackend::naive,
          ops::GemmBackend::tiled}) {
        const GemmBackendGuard guard(backend);
        std::uint64_t seed = 500;
        for (const bool ta : {false, true}) {
            for (const bool tb : {false, true}) {
                const Tensor a = random_tensor(stored({2}, 37, 129, ta), seed);
                const Tensor b =
                    random_tensor(stored({2}, 129, 65, tb), seed + 1);
                Tensor cpu = random_tensor({2, 37, 65}, seed + 2);
                Tensor gpu = cpu.to(Device::CUDA);
                ops::matmul_acc(cpu, a, b, ta, tb);
                ops::matmul_acc(gpu, a.to(Device::CUDA), b.to(Device::CUDA), ta,
                                tb);
                TITAN_EXPECT_PARITY(cpu, gpu, kMatmulAtol, kMatmulRtol);
                seed += 3;
            }
        }
    }
}

// cuBLAS and the tiled kernel agree on the same device inputs.
TITAN_TEST(MatmulParity, BackendsAgree) {
    TITAN_REQUIRE_CUDA();
    const Tensor a = random_tensor({2, 3, 37, 129}, 600).to(Device::CUDA);
    const Tensor b = random_tensor({2, 3, 65, 129}, 601).to(Device::CUDA);
    const Tensor reference = [&] {
        const GemmBackendGuard guard(ops::GemmBackend::cublas);
        return ops::matmul(a, b, false, true).to(Device::CPU);
    }();
    const GemmBackendGuard guard(ops::GemmBackend::tiled);
    const Tensor tiled = ops::matmul(a, b, false, true);
    TITAN_EXPECT_PARITY(reference, tiled, kMatmulAtol, kMatmulRtol);
}
