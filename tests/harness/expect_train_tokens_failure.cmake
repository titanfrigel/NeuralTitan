# Runs `titan train ARGS` (OUT_DIR is the run directory in ARGS), which must
# exit with 1, print a message matching PATTERN, and leave OUT_DIR uncreated.
file(REMOVE_RECURSE "${OUT_DIR}")
execute_process(
  COMMAND "${EXE}" ${ARGS}
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output)
message(STATUS "exit code: ${exit_code}\n${output}")
if(NOT exit_code STREQUAL "1")
  message(FATAL_ERROR "expected exit code 1, got '${exit_code}'")
endif()
if(NOT output MATCHES "${PATTERN}")
  message(FATAL_ERROR "the output does not match '${PATTERN}'")
endif()
if(EXISTS "${OUT_DIR}")
  message(FATAL_ERROR "${OUT_DIR} was created")
endif()
