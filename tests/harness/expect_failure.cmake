# Runs EXE with ARGS (a list; FILTER alone when ARGS is unset) and requires the
# exit code EXPECT_CODE (default 1) and output that matches PATTERN. Used by the harness self-tests: WILL_FAIL alone would also
# pass on a crash or launch failure. With NO_STDOUT, everything must go to
# stderr.
if(NOT DEFINED EXPECT_CODE)
  set(EXPECT_CODE 1)
endif()
if(NOT DEFINED ARGS)
  set(ARGS "${FILTER}")
endif()

if(NO_STDOUT)
  execute_process(
    COMMAND "${EXE}" ${ARGS}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE output)
  if(NOT stdout STREQUAL "")
    message(FATAL_ERROR "expected no stdout output, got '${stdout}'")
  endif()
else()
  execute_process(
    COMMAND "${EXE}" ${ARGS}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
endif()

message(STATUS "exit code: ${exit_code}\n${output}")

if(NOT exit_code STREQUAL EXPECT_CODE)
  message(FATAL_ERROR "expected exit code ${EXPECT_CODE}, got '${exit_code}'")
endif()
if(NOT output MATCHES "${PATTERN}")
  message(FATAL_ERROR "output does not match '${PATTERN}'")
endif()
