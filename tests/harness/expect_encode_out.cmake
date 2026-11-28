# Runs `titan tokenizer encode --tokenizer TOKENIZER --file CORPUS --out OUT`
# and requires: exit code 0; stdout of the form "tokens N / crc32 / bytes per
# token" with N > 0 and no ids; stdout equal to that of the same encode with
# --summary instead of --out (same count, CRC-32 and ratio); and OUT to be a
# token file of 16 header bytes plus 4 bytes per token.
file(REMOVE "${OUT}")
execute_process(
  COMMAND "${EXE}" tokenizer encode --tokenizer "${TOKENIZER}" --file "${CORPUS}"
    --out "${OUT}"
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE out_stdout
  ERROR_VARIABLE err)
if(NOT exit_code STREQUAL "0")
  message(FATAL_ERROR "encode --out exited with '${exit_code}': ${err}")
endif()
if(NOT out_stdout MATCHES "^tokens ([0-9]+)[\r\n]+crc32 0x[0-9a-f]+[\r\n]+bytes_per_token [0-9.]+[\r\n]*$")
  message(FATAL_ERROR "unexpected stdout: ${out_stdout}")
endif()
set(tokens "${CMAKE_MATCH_1}")
if(tokens EQUAL 0)
  message(FATAL_ERROR "no tokens were encoded")
endif()

execute_process(
  COMMAND "${EXE}" tokenizer encode --tokenizer "${TOKENIZER}" --file "${CORPUS}"
    --summary
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE summary_stdout
  ERROR_VARIABLE err)
if(NOT exit_code STREQUAL "0")
  message(FATAL_ERROR "encode --summary exited with '${exit_code}': ${err}")
endif()
if(NOT out_stdout STREQUAL summary_stdout)
  message(FATAL_ERROR
    "--out printed\n${out_stdout}\nbut --summary printed\n${summary_stdout}")
endif()

file(SIZE "${OUT}" size)
math(EXPR expected "16 + 4 * ${tokens}")
if(NOT size EQUAL expected)
  message(FATAL_ERROR "token file is ${size} bytes, expected ${expected}")
endif()
