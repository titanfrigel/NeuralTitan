#include "harness/test.hpp"

#include <stdexcept>

TITAN_TEST(Harness, ExpectNearWithinTolerance) {
    TITAN_EXPECT_NEAR(1.0f, 1.0005f, 1e-3f);
}

TITAN_TEST(Harness, ExpectThrowsMatchesType) {
    TITAN_EXPECT_THROWS(throw std::runtime_error("x"), std::runtime_error);
}

TITAN_TEST(Harness, SkipIsReported) { TITAN_SKIP("demo"); }
