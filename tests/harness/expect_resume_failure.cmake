# Trains CONFIG on DATA with --stop-at STOP_AT into a fresh OUT_DIR, replaces
# token INDEX of OUT_DIR/FILE with ID using TOOL (tests/tools/
# edit_token_file.cpp), then runs `titan train --resume OUT_DIR`. Requires exit
# code 1, output matching PATTERN and nothing on stdout.
file(REMOVE_RECURSE "${OUT_DIR}")
execute_process(
  COMMAND "${EXE}" train --config "${CONFIG}" --data "${DATA}"
    --out "${OUT_DIR}" --device cpu --stop-at ${STOP_AT}
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output)
if(NOT exit_code STREQUAL "0")
  message(FATAL_ERROR "the stopped run failed (${exit_code}):\n${output}")
endif()

execute_process(
  COMMAND "${TOOL}" "${OUT_DIR}/${FILE}" "${OUT_DIR}/${FILE}" ${INDEX} ${ID}
  RESULT_VARIABLE exit_code
  ERROR_VARIABLE output)
if(NOT exit_code STREQUAL "0")
  message(FATAL_ERROR "edit_token_file failed (${exit_code}):\n${output}")
endif()

execute_process(
  COMMAND "${EXE}" train --resume "${OUT_DIR}" --device cpu
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
message(STATUS "exit code: ${exit_code}\nstdout: ${stdout}\nstderr: ${stderr}")
if(NOT exit_code STREQUAL "1")
  message(FATAL_ERROR "expected exit code 1, got '${exit_code}'")
endif()
if(NOT stdout STREQUAL "")
  message(FATAL_ERROR "expected no stdout output")
endif()
if(NOT stderr MATCHES "${PATTERN}")
  message(FATAL_ERROR "stderr does not match '${PATTERN}'")
endif()
