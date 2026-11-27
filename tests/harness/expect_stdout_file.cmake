# Runs EXE with ARGS (a list) and requires exit code 0 and stdout equal, byte
# for byte apart from line endings, to the file EXPECTED. The stdout is
# written to OUT_FILE rather than to a CMake variable, which cannot hold the
# NUL bytes and invalid UTF-8 that raw token bytes may contain. The console
# stream turns each newline into CR LF on Windows, hence --ignore-eol.
get_filename_component(out_dir "${OUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${out_dir}")
execute_process(
  COMMAND "${EXE}" ${ARGS}
  RESULT_VARIABLE exit_code
  OUTPUT_FILE "${OUT_FILE}"
  ERROR_VARIABLE err)
message(STATUS "exit code: ${exit_code}\nstderr: ${err}")
if(NOT exit_code STREQUAL "0")
  message(FATAL_ERROR "expected exit code 0, got '${exit_code}'")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files --ignore-eol "${OUT_FILE}"
    "${EXPECTED}"
  RESULT_VARIABLE differ)
if(NOT differ STREQUAL "0")
  message(FATAL_ERROR "stdout (${OUT_FILE}) differs from ${EXPECTED}")
endif()
