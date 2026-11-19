# Runs the layer check on the violation fixture and requires a non-zero exit
# code plus one report line per planted violation. Checking the messages as
# well as the exit code keeps the check from passing on a crash or a typo.
execute_process(
  COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${SRC_DIR} -P ${CHECK_SCRIPT}
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output)

message(STATUS "exit code: ${exit_code}\n${output}")

if(exit_code STREQUAL "0")
  message(FATAL_ERROR "expected the layer check to fail on the violation fixture")
endif()

foreach(expected
    "nn may not include runtime"
    "ops/cpu may not include ops/cuda"
    "ops/cpu may not include cuda_runtime.h")
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "output does not contain '${expected}'")
  endif()
endforeach()
