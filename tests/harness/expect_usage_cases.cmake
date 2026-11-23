# Runs EXE once per case and requires exit code 2 and the usage text for each.
# CASES is a list of argument lists separated by the item "|".
set(case "")
list(APPEND CASES "|")
foreach(item IN LISTS CASES)
  if(NOT item STREQUAL "|")
    list(APPEND case "${item}")
    continue()
  endif()
  execute_process(
    COMMAND "${EXE}" ${case}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
  message(STATUS "titan ${case}\nexit code: ${exit_code}\n${output}")
  if(NOT exit_code STREQUAL "2")
    message(FATAL_ERROR "titan ${case}: expected exit code 2, got '${exit_code}'")
  endif()
  if(NOT output MATCHES "usage: titan")
    message(FATAL_ERROR "titan ${case}: no usage text")
  endif()
  set(case "")
endforeach()
