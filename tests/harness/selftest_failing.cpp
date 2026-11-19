#include "harness/test.hpp"

// Linked into harness_selftest_failing only. CTest runs it with WILL_FAIL to
// prove the runner reports a failing expectation through its exit code.
TITAN_TEST(SelfTest, AlwaysFails) {
    TITAN_EXPECT_EQ(1, 2);
}
