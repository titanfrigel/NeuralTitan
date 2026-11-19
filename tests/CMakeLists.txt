# Harness sources are shared by the real test executable and the self-test.
add_library(titan_test_harness STATIC harness/test.cpp)
target_include_directories(titan_test_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Test files are compiled straight into the executable, not into a static
# library: the linker would drop their static registrars otherwise.
# Adding a test file is a one-line change in this list.
add_executable(titan_tests harness/main.cpp)
target_sources(titan_tests PRIVATE
  core/harness_test.cpp
  core/error_test.cpp
  core/device_test.cpp
  core/allocator_test.cpp
  core/storage_test.cpp
  core/tensor_test.cpp
)
target_link_libraries(titan_tests PRIVATE titan_test_harness titan_core)
add_test(NAME unit COMMAND titan_tests)

add_executable(harness_selftest_failing harness/main.cpp harness/selftest_failing.cpp)
target_link_libraries(harness_selftest_failing PRIVATE titan_test_harness)

# PASS_REGULAR_EXPRESSION makes CTest ignore the exit code, so it must not be
# set on `unit`; the skip check is its own entry.
add_test(NAME harness_reports_skip COMMAND titan_tests Harness.SkipIsReported)
set_tests_properties(harness_reports_skip PROPERTIES
  PASS_REGULAR_EXPRESSION "SKIP Harness.SkipIsReported")

# Each self-test must exit with 1 and print the expected FAIL line.
function(add_selftest name filter pattern)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND}
      -DEXE=$<TARGET_FILE:harness_selftest_failing>
      -DFILTER=${filter}
      "-DPATTERN=${pattern}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_failure.cmake)
endfunction()

add_selftest(harness_detects_failure SelfTest.AlwaysFails
  "FAIL SelfTest.AlwaysFails .*1 == 2 \\(1 vs 2\\)")
add_selftest(harness_reports_unexpected_exception SelfTest.UnexpectedException
  "FAIL SelfTest.UnexpectedException .*unexpected exception: boom")
add_selftest(harness_detects_missing_throw SelfTest.ThrowsNothing
  "FAIL SelfTest.ThrowsNothing .*did not throw")
add_selftest(harness_detects_wrong_throw_type SelfTest.ThrowsWrongType
  "FAIL SelfTest.ThrowsWrongType .*did not throw")
