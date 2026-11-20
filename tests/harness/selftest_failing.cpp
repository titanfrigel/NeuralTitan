#include "harness/test.hpp"

#include <stdexcept>

// Linked into harness_selftest_failing only. CTest runs each test through
// expect_failure.cmake to prove the runner reports it as a failure.
TITAN_TEST(SelfTest, AlwaysFails) { TITAN_EXPECT_EQ(1, 2); }

// Adjacent floats must print as different numbers.
TITAN_TEST(SelfTest, FloatsDifferInLastDigit) {
    TITAN_EXPECT_EQ(1.0f, 1.0000001f);
}

TITAN_TEST(SelfTest, UnexpectedException) { throw std::runtime_error("boom"); }

TITAN_TEST(SelfTest, ThrowsNothing) {
    TITAN_EXPECT_THROWS((void)0, std::runtime_error);
}

TITAN_TEST(SelfTest, ThrowsWrongType) {
    TITAN_EXPECT_THROWS(throw std::logic_error("x"), std::runtime_error);
}
