#include "runtime/data_loader.hpp"

#include <string>
#include <utility>

#include "core/error.hpp"

namespace titan::runtime {

DataLoader::DataLoader(std::vector<std::int32_t> tokens, std::int64_t batch,
                       std::int64_t seq, std::uint64_t seed)
    : tokens_(std::move(tokens)), batch_(batch), seq_(seq), rng_(seed) {
    TITAN_CHECK(batch >= 1,
                "DataLoader: batch must be >= 1, got " + std::to_string(batch));
    TITAN_CHECK(seq >= 1,
                "DataLoader: seq must be >= 1, got " + std::to_string(seq));
    const auto n = static_cast<std::int64_t>(tokens_.size());
    TITAN_CHECK(n >= seq + 2, "DataLoader: " + std::to_string(n) +
                                  " tokens, but seq " + std::to_string(seq) +
                                  " needs at least " + std::to_string(seq + 2));
}

void DataLoader::set_sequential(bool sequential) {
    sequential_ = sequential;
    cursor_ = 0;
}

DataLoader::Batch DataLoader::next() {
    const auto n = static_cast<std::int64_t>(tokens_.size());
    Batch b;
    b.inputs.reserve(static_cast<std::size_t>(batch_ * seq_));
    b.targets.reserve(static_cast<std::size_t>(batch_ * seq_));
    for (std::int64_t row = 0; row < batch_; ++row) {
        std::int64_t offset = 0;
        if (sequential_) {
            if (cursor_ + seq_ + 1 > n) {
                cursor_ = 0;
            }
            offset = cursor_;
            cursor_ += seq_;
        } else {
            offset = rng_.randint(n - seq_);
        }
        const auto first = tokens_.begin() + offset;
        b.inputs.insert(b.inputs.end(), first, first + seq_);
        b.targets.insert(b.targets.end(), first + 1, first + 1 + seq_);
    }
    return b;
}

}  // namespace titan::runtime
