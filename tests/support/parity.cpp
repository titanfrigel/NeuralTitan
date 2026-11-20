#include "support/parity.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "core/rng.hpp"

namespace titan::test {

Tensor random_tensor(Shape s, std::uint64_t seed) {
    Rng rng(seed);
    Tensor t = Tensor::empty(std::move(s), Device::CPU);
    rng.fill_normal(t, 0.0f, 1.0f);
    return t;
}

ParityResult check_parity(const Tensor& cpu, const Tensor& cuda, double atol,
                          double rtol) {
    if (cpu.device() != Device::CPU || cuda.device() != Device::CUDA) {
        return {false, std::string("expected a cpu and a cuda tensor, got ") +
                           std::string(to_string(cpu.device())) + " and " +
                           std::string(to_string(cuda.device()))};
    }
    if (cpu.shape() != cuda.shape()) {
        return {false, "shape mismatch, " + to_string(cpu.shape()) + " vs " +
                           to_string(cuda.shape())};
    }

    const std::vector<float> c = cpu.to_vector();
    const std::vector<float> g = cuda.to_vector();
    std::size_t worst = 0;
    // How far the worst element exceeds its bound; infinity for a NaN.
    double worst_excess = -std::numeric_limits<double>::infinity();
    std::size_t failures = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const double cv = c[i];
        const double gv = g[i];
        const double bound = atol + rtol * std::fabs(cv);
        const double diff = std::fabs(cv - gv);
        // Written as !(diff <= bound) so that a NaN counts as a failure.
        if (!(diff <= bound)) {
            ++failures;
            const double excess = std::isnan(diff)
                                      ? std::numeric_limits<double>::infinity()
                                      : diff - bound;
            if (excess > worst_excess) {
                worst_excess = excess;
                worst = i;
            }
        }
    }
    if (failures == 0) return {true, ""};

    std::ostringstream os;
    os << std::setprecision(std::numeric_limits<float>::max_digits10)
       << failures << " of " << c.size() << " elements outside atol " << atol
       << " + rtol " << rtol << " * |cpu|; worst at index " << worst << ": cpu "
       << c[worst] << " vs cuda " << g[worst];
    return {false, os.str()};
}

}  // namespace titan::test
