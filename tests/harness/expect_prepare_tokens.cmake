# Prepares the token files of the cli_train_tokens_* tests in DIR with
# `titan tokenizer encode --out` and TOKENIZER: train.bin and val.bin (both
# from CORPUS), short.bin (one token) and train_oob.bin (train.bin with token 5 set
# to 512, outside the 512-token vocabulary, by TOOL).
file(REMOVE_RECURSE "${DIR}")

function(run_tool)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
  if(NOT exit_code STREQUAL "0")
    message(FATAL_ERROR "${ARGN} exited with '${exit_code}':\n${output}")
  endif()
endfunction()

run_tool("${EXE}" tokenizer encode --tokenizer "${TOKENIZER}" --file "${CORPUS}"
  --out "${DIR}/train.bin")
run_tool("${EXE}" tokenizer encode --tokenizer "${TOKENIZER}" --file "${CORPUS}"
  --out "${DIR}/val.bin")
run_tool("${EXE}" tokenizer encode --tokenizer "${TOKENIZER}" --text a
  --out "${DIR}/short.bin")
run_tool("${TOOL}" "${DIR}/train.bin" "${DIR}/train_oob.bin" 5 512)
