# Runs `titan generate --checkpoint CHECKPOINT --prompt PROMPT --max-new
# MAX_NEW` with the extra arguments ARGS and requires exit code 0, stdout that
# is the prompt followed by MAX_NEW characters and a newline (the fixture
# vocabulary is ASCII, so characters are bytes), and a tokens/s line on
# stderr. With COMPARE_ARGS, runs again with those extra arguments instead
# and requires the same stdout. CUDA_SKIP makes a missing GPU print
# "no CUDA device available" and succeed, for SKIP_REGULAR_EXPRESSION.
function(run_generate out_var)
  execute_process(
    COMMAND "${EXE}" generate --checkpoint "${CHECKPOINT}" --prompt "${PROMPT}"
      --max-new ${MAX_NEW} ${ARGN}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  message(STATUS "generate ${ARGN}\nexit code: ${exit_code}\n"
    "stdout: ${out}\nstderr: ${err}")
  if(CUDA_SKIP AND err MATCHES "no CUDA device available")
    message(STATUS "no CUDA device available")
    return()
  endif()
  if(NOT exit_code STREQUAL "0")
    message(FATAL_ERROR "expected exit code 0, got '${exit_code}'")
  endif()
  # The console stream turns each newline into CR LF on Windows.
  string(REPLACE "\r\n" "\n" out "${out}")
  string(LENGTH "${PROMPT}" prompt_length)
  string(SUBSTRING "${out}" 0 ${prompt_length} head)
  if(NOT head STREQUAL PROMPT)
    message(FATAL_ERROR "stdout does not start with the prompt")
  endif()
  string(LENGTH "${out}" out_length)
  math(EXPR want_length "${prompt_length} + ${MAX_NEW} + 1")
  if(NOT out_length EQUAL want_length)
    message(FATAL_ERROR "stdout has ${out_length} bytes, want ${want_length}")
  endif()
  if(NOT err MATCHES "tokens/s [0-9.]+")
    message(FATAL_ERROR "stderr lacks 'tokens/s'")
  endif()
  set(${out_var} "${out}" PARENT_SCOPE)
endfunction()

run_generate(first ${ARGS})
if(DEFINED COMPARE_ARGS)
  run_generate(second ${COMPARE_ARGS})
  if(NOT first STREQUAL second)
    message(FATAL_ERROR "stdout differs:\n${first}\nvs\n${second}")
  endif()
endif()
