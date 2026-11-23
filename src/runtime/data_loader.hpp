#pragma once

#include <cstdint>
#include <vector>

#include "core/rng.hpp"

namespace titan::runtime {

// Draws training windows from one token stream. Each row of a batch is a
// window of `seq` consecutive tokens (the inputs) and the same window shifted
// one token to the right (the targets).
class DataLoader {
public:
    struct Batch {
        // Both hold batch * seq ids, row-major (batch, seq).
        std::vector<std::int32_t> inputs, targets;
    };

    // Throws titan::Error unless batch >= 1, seq >= 1 and
    // tokens.size() >= seq + 2 (a window needs seq + 1 tokens, and offsets are
    // drawn from at least two positions).
    DataLoader(std::vector<std::int32_t> tokens, std::int64_t batch,
               std::int64_t seq, std::uint64_t seed);

    // `batch` windows at offsets drawn uniformly from [0, n - seq - 1] with
    // Rng::randint, so a draw is fully determined by the seed.
    Batch next();

    // Test seam. When enabled, next() ignores the Rng and yields consecutive
    // windows starting at offsets 0, seq, 2*seq, ..., wrapping to offset 0
    // when the next window would overrun the stream. It lets a test feed two
    // loaders with different batch sizes the same windows.
    void set_sequential(bool sequential);

    // The generator that draws the offsets, so a checkpoint can save and
    // restore it (Rng::save_state, Rng::load_state).
    const Rng& rng() const { return rng_; }
    Rng& rng() { return rng_; }

private:
    std::vector<std::int32_t> tokens_;
    std::int64_t batch_, seq_;
    Rng rng_;
    bool sequential_ = false;
    std::int64_t cursor_ = 0;
};

}  // namespace titan::runtime
