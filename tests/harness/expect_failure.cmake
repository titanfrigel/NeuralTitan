# Runs EXE with ARGS (a list; FILTER alone when ARGS is unset) and requires both a non-zero exit code and output that
# matches PATTERN. Used by the harness self-tests: WILL_FAIL alone would also
# pass on a crash or launch failure.
if(NOT DEFINED ARGS)
  set(ARGS "${FILTER}")
endif()

execute_process(
  COMMAND "${EXE}" ${ARGS}
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output)

message(STATUS "exit code: ${exit_code}\n${output}")

if(exit_code STREQUAL "0")
  message(FATAL_ERROR "expected a non-zero exit code")
endif()
if(NOT exit_code MATCHES "^[0-9]+$" OR exit_code GREATER 1)
  message(FATAL_ERROR "expected exit code 1 (test failure), got '${exit_code}'")
endif()
if(NOT output MATCHES "${PATTERN}")
  message(FATAL_ERROR "output does not match '${PATTERN}'")
endif()
