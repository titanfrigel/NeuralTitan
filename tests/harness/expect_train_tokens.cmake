# Trains CONFIG for 6 steps from the token files TRAIN_TOKENS and VAL_TOKENS
# with TOKENIZER into OUT_DIR/a on the CPU, and requires exit code 0, the
# files latest.ckpt, train.bin, val.bin and sample.txt, and train.bin and
# val.bin to be byte-identical to the inputs.
#
# With MODE=resume it also trains the same run with --stop-at 3 into OUT_DIR/b,
# continues it with --resume, and requires the final latest.ckpt of both runs
# to be byte-identical.
file(REMOVE_RECURSE "${OUT_DIR}")

function(run_titan)
  execute_process(
    COMMAND "${EXE}" ${ARGN}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
  message(STATUS "titan ${ARGN}\nexit code: ${exit_code}\n${output}")
  if(NOT exit_code STREQUAL "0")
    message(FATAL_ERROR "titan ${ARGN} exited with '${exit_code}'")
  endif()
endfunction()

set(common --config "${CONFIG}" --train-tokens "${TRAIN_TOKENS}"
  --val-tokens "${VAL_TOKENS}" --tokenizer "${TOKENIZER}" --steps 6
  --device cpu)

run_titan(train ${common} --out "${OUT_DIR}/a")
foreach(name latest.ckpt train.bin val.bin sample.txt)
  if(NOT EXISTS "${OUT_DIR}/a/${name}")
    message(FATAL_ERROR "the run wrote no ${name}")
  endif()
endforeach()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E compare_files "${TRAIN_TOKENS}" "${OUT_DIR}/a/train.bin"
  RESULT_VARIABLE differs)
if(NOT differs STREQUAL "0")
  message(FATAL_ERROR "train.bin differs from ${TRAIN_TOKENS}")
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E compare_files "${VAL_TOKENS}" "${OUT_DIR}/a/val.bin"
  RESULT_VARIABLE differs)
if(NOT differs STREQUAL "0")
  message(FATAL_ERROR "val.bin differs from ${VAL_TOKENS}")
endif()

if(MODE STREQUAL "resume")
  run_titan(train ${common} --out "${OUT_DIR}/b" --stop-at 3)
  run_titan(train --resume "${OUT_DIR}/b" --device cpu)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files
      "${OUT_DIR}/a/latest.ckpt" "${OUT_DIR}/b/latest.ckpt"
    RESULT_VARIABLE differs)
  if(NOT differs STREQUAL "0")
    message(FATAL_ERROR "the resumed latest.ckpt differs from the uninterrupted one")
  endif()
endif()
