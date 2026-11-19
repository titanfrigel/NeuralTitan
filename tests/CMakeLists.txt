# Harness sources are shared by the real test executable and the self-test.
add_library(titan_test_harness STATIC harness/test.cpp)
target_include_directories(titan_test_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Test files are compiled straight into the executable, not into a static
# library: the linker would drop their static registrars otherwise.
# Adding a test file is a one-line change in this list.
add_executable(titan_tests harness/main.cpp)
target_sources(titan_tests PRIVATE
  core/harness_test.cpp
)
target_link_libraries(titan_tests PRIVATE titan_test_harness)
add_test(NAME unit COMMAND titan_tests)

add_executable(harness_selftest_failing harness/main.cpp harness/selftest_failing.cpp)
target_link_libraries(harness_selftest_failing PRIVATE titan_test_harness)
add_test(NAME harness_detects_failure COMMAND harness_selftest_failing)
set_tests_properties(harness_detects_failure PROPERTIES WILL_FAIL TRUE)
