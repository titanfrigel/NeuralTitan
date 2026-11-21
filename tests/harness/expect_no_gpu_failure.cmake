# Runs `titan train ... --device cuda` with every CUDA device hidden
# (CUDA_VISIBLE_DEVICES=-1, so the test also runs on a machine with a GPU) and
# requires exit code 1, the "error: " prefix, the no-device message and that
# OUT_DIR was never created.
set(ENV{CUDA_VISIBLE_DEVICES} "-1")
file(REMOVE_RECURSE "${OUT_DIR}")

execute_process(
  COMMAND "${EXE}" train --config "${CONFIG}" --data "${DATA}"
    --out "${OUT_DIR}" --device cuda
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output)

message(STATUS "exit code: ${exit_code}\n${output}")

if(NOT exit_code STREQUAL "1")
  message(FATAL_ERROR "expected exit code 1, got '${exit_code}'")
endif()
if(NOT output MATCHES "error: no CUDA device available")
  message(FATAL_ERROR "output lacks 'error: no CUDA device available'")
endif()
if(EXISTS "${OUT_DIR}")
  message(FATAL_ERROR "${OUT_DIR} was created before the device check")
endif()
