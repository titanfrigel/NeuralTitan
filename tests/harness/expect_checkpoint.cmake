# Trains CONFIG on DATA on the CPU into a fresh OUT_DIR, with the extra train
# arguments TRAIN_ARGS (a list, optional), and requires exit code 0 and
# OUT_DIR/latest.ckpt. With GENERATE_ARGS (a list), then runs
# `generate --checkpoint OUT_DIR/latest.ckpt GENERATE_ARGS` and requires exit
# code 0.
file(REMOVE_RECURSE "${OUT_DIR}")
execute_process(
  COMMAND "${EXE}" train --config "${CONFIG}" --data "${DATA}"
    --out "${OUT_DIR}" --device cpu ${TRAIN_ARGS}
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output)
message(STATUS "exit code: ${exit_code}\n${output}")
if(NOT exit_code STREQUAL "0")
  message(FATAL_ERROR "expected exit code 0, got '${exit_code}'")
endif()
if(NOT EXISTS "${OUT_DIR}/latest.ckpt")
  message(FATAL_ERROR "${OUT_DIR}/latest.ckpt was not written")
endif()

if(DEFINED GENERATE_ARGS)
  execute_process(
    COMMAND "${EXE}" generate --checkpoint "${OUT_DIR}/latest.ckpt"
      ${GENERATE_ARGS}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
  message(STATUS "generate exit code: ${exit_code}\n${output}")
  if(NOT exit_code STREQUAL "0")
    message(FATAL_ERROR "generate: expected exit code 0, got '${exit_code}'")
  endif()
endif()
