# Runs EXE with ARGS (a list) and requires exit code 0 and stdout that is
# valid UTF-8 and starts with PREFIX. CMake regular expressions work on
# bytes but cannot name bytes above 0x7F, so stdout is checked as a hex
# string: each alternative of `sequence` below consumes one whole UTF-8
# sequence (RFC 3629, Table 3-7 of the Unicode standard) as hex digit pairs,
# so the string matches only when every byte belongs to a valid sequence.
execute_process(
  COMMAND "${EXE}" ${ARGS}
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
string(HEX "${stdout}" hex)
message(STATUS "exit code: ${exit_code}\nstdout (hex): ${hex}\n"
  "stderr: ${stderr}")
if(NOT exit_code STREQUAL "0")
  message(FATAL_ERROR "expected exit code 0, got '${exit_code}'")
endif()

set(x "[0-9a-f]")
set(cont "[89ab]${x}")  # a continuation byte, 80..BF
set(sequence
  "[0-7]${x}"                             # 00..7F
  "(c[2-9a-f]|d${x})${cont}"              # C2..DF, one continuation
  "e0[ab]${x}${cont}"                     # E0 A0..BF: no overlong form
  "(e[1-9a-c]|e[ef])${cont}${cont}"       # E1..EC, EE..EF
  "ed[89]${x}${cont}"                     # ED 80..9F: no surrogate
  "f0[9ab]${x}${cont}${cont}"             # F0 90..BF: no overlong form
  "f[1-3]${cont}${cont}${cont}"           # F1..F3
  "f48${x}${cont}${cont}")                # F4 80..8F: at most U+10FFFF
list(JOIN sequence "|" alternatives)
if(NOT hex MATCHES "^(${alternatives})*$")
  message(FATAL_ERROR "stdout is not valid UTF-8")
endif()

string(LENGTH "${PREFIX}" prefix_length)
string(SUBSTRING "${stdout}" 0 ${prefix_length} head)
if(NOT head STREQUAL PREFIX)
  message(FATAL_ERROR "stdout does not start with '${PREFIX}'")
endif()
