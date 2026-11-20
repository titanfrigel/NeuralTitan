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
  core/rng_test.cpp
  core/utf8_test.cpp
  core/parallel_test.cpp
  core/cuda_toolchain_test.cu
  support/gradcheck.cpp
  support/gradcheck_test.cpp
  ops/elementwise_test.cpp
  ops/reduce_test.cpp
  ops/matmul_test.cpp
  ops/softmax_test.cpp
  ops/norm_test.cpp
  ops/rope_test.cpp
  ops/embedding_test.cpp
  ops/cross_entropy_test.cpp
  ops/adamw_test.cpp
  nn/linear_test.cpp
  nn/norm_module_test.cpp
  nn/mlp_test.cpp
  nn/embedding_module_test.cpp
  optim/optim_test.cpp
  io/json_test.cpp
  io/token_file_test.cpp
)
# The CUDA tests call the runtime directly, so they link cudart themselves
# instead of relying on titan_core, which keeps it private.
target_link_libraries(titan_tests PRIVATE titan_test_harness titan_io titan_optim titan_nn titan_ops titan_core CUDA::cudart)
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
add_selftest(harness_prints_float_digits SelfTest.FloatsDifferInLastDigit
  "FAIL SelfTest.FloatsDifferInLastDigit .*\\(1 vs 1\\.00000012\\)")
add_selftest(harness_reports_unexpected_exception SelfTest.UnexpectedException
  "FAIL SelfTest.UnexpectedException .*unexpected exception: boom")
add_selftest(harness_detects_missing_throw SelfTest.ThrowsNothing
  "FAIL SelfTest.ThrowsNothing .*did not throw")
add_selftest(harness_detects_wrong_throw_type SelfTest.ThrowsWrongType
  "FAIL SelfTest.ThrowsWrongType .*did not throw")

# Layer include rules (docs/architecture/overview.md section 10). The fixtures
# under layer_check/ are only scanned as text and belong to no target.
set(LAYER_CHECK_SCRIPT ${PROJECT_SOURCE_DIR}/scripts/check_layers.cmake)
add_test(NAME layers_clean_fixture
  COMMAND ${CMAKE_COMMAND}
    -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/layer_check/clean/src
    -P ${LAYER_CHECK_SCRIPT})
add_test(NAME layers_violation_fixture
  COMMAND ${CMAKE_COMMAND}
    -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/layer_check/violation/src
    -DCHECK_SCRIPT=${LAYER_CHECK_SCRIPT}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/layer_check/expect_violations.cmake)
add_test(NAME layers_repo
  COMMAND ${CMAKE_COMMAND}
    -DSRC_DIR=${PROJECT_SOURCE_DIR}/src
    -P ${LAYER_CHECK_SCRIPT})

# CLI smoke tests. $<TARGET_FILE:titan> keeps these independent of the order
# in which apps/ and tests/ are added.
# The dots of the version are escaped so they only match literal dots.
string(REPLACE "." "\\." _titan_version_re "${PROJECT_VERSION}")
add_test(NAME cli_version COMMAND $<TARGET_FILE:titan> --version)
set_tests_properties(cli_version PROPERTIES
  PASS_REGULAR_EXPRESSION "^titan ${_titan_version_re}")

add_test(NAME cli_info COMMAND $<TARGET_FILE:titan> info)
set_tests_properties(cli_info PROPERTIES PASS_REGULAR_EXPRESSION "CUDA: ")

add_test(NAME cli_unknown_command COMMAND $<TARGET_FILE:titan> bogus)
set_tests_properties(cli_unknown_command PROPERTIES WILL_FAIL TRUE)
