# Checks that an interrupted and resumed run ends exactly where an
# uninterrupted one does. Run A trains CONFIG on DATA for STEPS steps in
# OUT_DIR/a. Run B trains the same with --stop-at STOP_AT in OUT_DIR/b, which
# must stop there and leave latest.ckpt, then continues with --resume. The
# final "eval step STEPS val_loss" lines of A and of the resumed B must be
# equal.
file(REMOVE_RECURSE "${OUT_DIR}")

function(run_titan out_var)
  execute_process(
    COMMAND "${EXE}" ${ARGN}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
  message(STATUS "titan ${ARGN}\nexit code: ${exit_code}\n${output}")
  if(NOT exit_code STREQUAL "0")
    message(FATAL_ERROR "titan ${ARGN} exited with '${exit_code}'")
  endif()
  set(${out_var} "${output}" PARENT_SCOPE)
endfunction()

function(final_eval out_var output)
  string(REGEX MATCH "eval step ${STEPS} val_loss [0-9.]+" line "${output}")
  if(line STREQUAL "")
    message(FATAL_ERROR "no 'eval step ${STEPS} val_loss' line")
  endif()
  set(${out_var} "${line}" PARENT_SCOPE)
endfunction()

set(common --config "${CONFIG}" --data "${DATA}" --steps ${STEPS} --device cpu)

run_titan(a train ${common} --out "${OUT_DIR}/a")
final_eval(want "${a}")

run_titan(b1 train ${common} --out "${OUT_DIR}/b" --stop-at ${STOP_AT})
if(NOT b1 MATCHES "stopped after step ${STOP_AT}")
  message(FATAL_ERROR "the run did not stop after step ${STOP_AT}")
endif()
if(NOT EXISTS "${OUT_DIR}/b/latest.ckpt")
  message(FATAL_ERROR "the stopped run left no latest.ckpt")
endif()

run_titan(b2 train --resume "${OUT_DIR}/b" --device cpu)
final_eval(got "${b2}")

if(NOT got STREQUAL want)
  message(FATAL_ERROR "resumed '${got}' differs from uninterrupted '${want}'")
endif()
