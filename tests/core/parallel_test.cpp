#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/error.hpp"
#include "core/parallel.hpp"
#include "harness/test.hpp"

TITAN_TEST(Parallel, CoversRangeOnce) {
    const int previous = titan::num_threads();
    titan::set_num_threads(4);
    std::vector<std::atomic<int>> hits(1003);
    titan::parallel_for(0, 1003, [&](std::int64_t b, std::int64_t e) {
        for (std::int64_t i = b; i < e; ++i) {
            hits[static_cast<std::size_t>(i)]++;
        }
    });
    bool once = true;
    for (const auto& h : hits) {
        once = once && h.load() == 1;
    }
    TITAN_EXPECT(once);
    titan::set_num_threads(previous);
}

TITAN_TEST(Parallel, PropagatesException) {
    const int previous = titan::num_threads();
    titan::set_num_threads(4);
    auto body = [](std::int64_t b, std::int64_t e) {
        if (b <= 500 && 500 < e) {
            throw std::runtime_error("boom");
        }
    };
    TITAN_EXPECT_THROWS(titan::parallel_for(0, 1003, body), std::runtime_error);
    titan::set_num_threads(previous);
}

TITAN_TEST(Parallel, SetNumThreadsRejectsZero) {
    TITAN_EXPECT_THROWS(titan::set_num_threads(0), titan::Error);
    TITAN_EXPECT(titan::num_threads() >= 1);
}
