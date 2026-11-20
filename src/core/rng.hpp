#pragma once

#include <cstdint>
#include <random>

#include "core/tensor.hpp"

namespace titan {

// Seeded random number generator. One seed drives initialization, data order
// and sampling. The engine (std::mt19937_64) has a sequence fixed by the
// standard; its distributions do not, so the transforms below are our own and
// give the same values for a given standard library and platform (they use
// std::log, std::sin and std::cos).
class Rng {
public:
    explicit Rng(std::uint64_t seed) : engine_(seed) {}

    std::uint64_t next_u64() { return engine_(); }

    // Uniform in [0, 1): the top 24 bits of a draw times 2^-24, so every
    // value is exactly representable as a float.
    float uniform();

    // Standard normal by the Box-Muller transform, computed in double. Each
    // pair of uniforms yields two normals; the second is cached.
    float normal();

    // Uniform integer in [0, n) by rejection sampling (no modulo bias).
    // Throws if n <= 0.
    std::int64_t randint(std::int64_t n);

    // Fills a CPU tensor with mean + stddev * normal(). Throws for a CUDA
    // tensor.
    void fill_normal(Tensor& t, float mean, float stddev);

private:
    std::mt19937_64 engine_;
    double cached_normal_ = 0.0;
    bool has_cached_normal_ = false;
};

}  // namespace titan
