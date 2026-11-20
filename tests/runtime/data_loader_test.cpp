#include <cstdint>
#include <vector>

#include "core/error.hpp"
#include "harness/test.hpp"
#include "runtime/data_loader.hpp"

using titan::runtime::DataLoader;

namespace {

std::vector<std::int32_t> counting_tokens(std::int32_t n) {
    std::vector<std::int32_t> t(static_cast<std::size_t>(n));
    for (std::int32_t i = 0; i < n; ++i) {
        t[static_cast<std::size_t>(i)] = i;
    }
    return t;
}

}  // namespace

TITAN_TEST(DataLoader, TargetsAreShiftedInputs) {
    // With tokens 0..n-1 a window starting at o is o, o+1, ... so the targets
    // are the inputs plus one and every window stays inside the stream.
    const std::int64_t batch = 3;
    const std::int64_t seq = 5;
    DataLoader loader(counting_tokens(40), batch, seq, 7);
    for (int i = 0; i < 50; ++i) {
        const DataLoader::Batch b = loader.next();
        TITAN_EXPECT_EQ(b.inputs.size(), std::size_t{batch * seq});
        TITAN_EXPECT_EQ(b.targets.size(), std::size_t{batch * seq});
        for (std::int64_t row = 0; row < batch; ++row) {
            for (std::int64_t t = 0; t < seq; ++t) {
                const auto k = static_cast<std::size_t>(row * seq + t);
                TITAN_EXPECT_EQ(b.targets[k], b.inputs[k] + 1);
                if (t > 0) {
                    TITAN_EXPECT_EQ(b.inputs[k], b.inputs[k - 1] + 1);
                }
                TITAN_EXPECT(b.targets[k] <= 39);
            }
        }
    }
}

TITAN_TEST(DataLoader, SameSeedSameBatches) {
    DataLoader a(counting_tokens(100), 4, 8, 123);
    DataLoader b(counting_tokens(100), 4, 8, 123);
    DataLoader c(counting_tokens(100), 4, 8, 124);
    bool differs = false;
    for (int i = 0; i < 5; ++i) {
        const DataLoader::Batch x = a.next();
        const DataLoader::Batch y = b.next();
        const DataLoader::Batch z = c.next();
        TITAN_EXPECT(x.inputs == y.inputs);
        TITAN_EXPECT(x.targets == y.targets);
        differs = differs || x.inputs != z.inputs;
    }
    TITAN_EXPECT(differs);
}

TITAN_TEST(DataLoader, TooShortThrows) {
    // seq 8 needs at least 10 tokens.
    TITAN_EXPECT_THROWS(DataLoader(counting_tokens(9), 2, 8, 1), titan::Error);
    DataLoader ok(counting_tokens(10), 2, 8, 1);
    TITAN_EXPECT_EQ(ok.next().inputs.size(), std::size_t{16});
    TITAN_EXPECT_THROWS(DataLoader(counting_tokens(10), 0, 8, 1), titan::Error);
    TITAN_EXPECT_THROWS(DataLoader(counting_tokens(10), 2, 0, 1), titan::Error);
}

TITAN_TEST(DataLoader, SequentialYieldsConsecutiveWindowsAndWraps) {
    // 11 tokens, seq 4: windows at 0 and 4; a window at 8 would overrun.
    DataLoader loader(counting_tokens(11), 3, 4, 1);
    loader.set_sequential(true);
    const DataLoader::Batch b = loader.next();
    const std::vector<std::int32_t> expected = {0, 1, 2, 3, 4, 5,
                                                6, 7, 0, 1, 2, 3};
    TITAN_EXPECT(b.inputs == expected);
    TITAN_EXPECT_EQ(b.targets[0], 1);
}
