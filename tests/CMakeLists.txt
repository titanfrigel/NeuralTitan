# Harness sources are shared by the real test executable and the self-test.
add_library(titan_test_harness STATIC harness/test.cpp)
target_include_directories(titan_test_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Test files are compiled straight into the executable, not into a static
# library: the linker would drop their static registrars otherwise.
# Adding a test file is a one-line change in this list.
add_executable(titan_tests harness/main.cpp)
target_sources(titan_tests PRIVATE
  core/harness_test.cpp
  core/error_test.cpp
  core/device_test.cpp
  core/allocator_test.cpp
  core/storage_test.cpp
  core/tensor_test.cpp
  core/rng_test.cpp
  core/utf8_test.cpp
  core/parallel_test.cpp
  core/cuda_toolchain_test.cu
  support/gradcheck.cpp
  support/gradcheck_test.cpp
  support/parity.cpp
  support/parity_test.cpp
  ops/elementwise_test.cpp
  ops/reduce_test.cpp
  ops/matmul_test.cpp
  ops/softmax_test.cpp
  ops/norm_test.cpp
  ops/rope_test.cpp
  ops/embedding_test.cpp
  ops/cross_entropy_test.cpp
  ops/adamw_test.cpp
  ops/layout_test.cpp
  ops/cuda/elementwise_parity_test.cpp
  ops/cuda/reduce_parity_test.cpp
  ops/cuda/softmax_parity_test.cpp
  ops/cuda/norm_parity_test.cpp
  ops/cuda/matmul_parity_test.cpp
  ops/cuda/rope_parity_test.cpp
  ops/cuda/layout_parity_test.cpp
  ops/cuda/embedding_parity_test.cpp
  ops/cuda/cross_entropy_parity_test.cpp
  ops/cuda/adamw_parity_test.cpp
  nn/linear_test.cpp
  nn/norm_module_test.cpp
  nn/mlp_test.cpp
  nn/embedding_module_test.cpp
  nn/attention_test.cpp
  nn/block_test.cpp
  nn/module_parity_test.cpp
  nn/kv_cache_test.cpp
  optim/optim_test.cpp
  optim/adamw_test.cpp
  io/json_test.cpp
  io/token_file_test.cpp
  io/checkpoint_file_test.cpp
  tokenizer/char_tokenizer_test.cpp
  model/config_test.cpp
  model/model_test.cpp
  model/model_cuda_test.cpp
  runtime/data_loader_test.cpp
  runtime/train_config_test.cpp
  runtime/trainer_test.cpp
  runtime/checkpoint_test.cpp
  runtime/sampler_test.cpp
  runtime/generator_test.cpp
)
# The CUDA tests call the runtime directly, so they link cudart themselves
# instead of relying on titan_core, which keeps it private.
target_link_libraries(titan_tests PRIVATE titan_test_harness titan_runtime titan_model titan_tokenizer titan_io titan_optim titan_nn titan_ops titan_core CUDA::cudart)
add_test(NAME unit COMMAND titan_tests)

add_executable(harness_selftest_failing harness/main.cpp harness/selftest_failing.cpp)
target_link_libraries(harness_selftest_failing PRIVATE titan_test_harness)

# PASS_REGULAR_EXPRESSION makes CTest ignore the exit code, so it must not be
# set on `unit`; the skip check is its own entry.
add_test(NAME harness_reports_skip COMMAND titan_tests Harness.SkipIsReported)
set_tests_properties(harness_reports_skip PROPERTIES
  PASS_REGULAR_EXPRESSION "SKIP Harness.SkipIsReported")

# Each self-test must exit with 1 and print the expected FAIL line.
function(add_selftest name filter pattern)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND}
      -DEXE=$<TARGET_FILE:harness_selftest_failing>
      -DFILTER=${filter}
      "-DPATTERN=${pattern}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_failure.cmake)
endfunction()

add_selftest(harness_detects_failure SelfTest.AlwaysFails
  "FAIL SelfTest.AlwaysFails .*1 == 2 \\(1 vs 2\\)")
add_selftest(harness_prints_float_digits SelfTest.FloatsDifferInLastDigit
  "FAIL SelfTest.FloatsDifferInLastDigit .*\\(1 vs 1\\.00000012\\)")
add_selftest(harness_reports_unexpected_exception SelfTest.UnexpectedException
  "FAIL SelfTest.UnexpectedException .*unexpected exception: boom")
add_selftest(harness_detects_missing_throw SelfTest.ThrowsNothing
  "FAIL SelfTest.ThrowsNothing .*did not throw")
add_selftest(harness_detects_wrong_throw_type SelfTest.ThrowsWrongType
  "FAIL SelfTest.ThrowsWrongType .*did not throw")

# Layer include rules (docs/architecture/overview.md section 10). The fixtures
# under layer_check/ are only scanned as text and belong to no target.
set(LAYER_CHECK_SCRIPT ${PROJECT_SOURCE_DIR}/scripts/check_layers.cmake)
add_test(NAME layers_clean_fixture
  COMMAND ${CMAKE_COMMAND}
    -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/layer_check/clean/src
    -P ${LAYER_CHECK_SCRIPT})
add_test(NAME layers_violation_fixture
  COMMAND ${CMAKE_COMMAND}
    -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/layer_check/violation/src
    -DCHECK_SCRIPT=${LAYER_CHECK_SCRIPT}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/layer_check/expect_violations.cmake)
add_test(NAME layers_repo
  COMMAND ${CMAKE_COMMAND}
    -DSRC_DIR=${PROJECT_SOURCE_DIR}/src
    -P ${LAYER_CHECK_SCRIPT})

# CLI smoke tests. $<TARGET_FILE:titan> keeps these independent of the order
# in which apps/ and tests/ are added.
# The dots of the version are escaped so they only match literal dots.
string(REPLACE "." "\\." _titan_version_re "${PROJECT_VERSION}")
add_test(NAME cli_version COMMAND $<TARGET_FILE:titan> --version)
set_tests_properties(cli_version PROPERTIES
  PASS_REGULAR_EXPRESSION "^titan ${_titan_version_re}")

add_test(NAME cli_info COMMAND $<TARGET_FILE:titan> info)
set_tests_properties(cli_info PROPERTIES PASS_REGULAR_EXPRESSION "CUDA: ")

add_test(NAME cli_unknown_command COMMAND $<TARGET_FILE:titan> bogus)
set_tests_properties(cli_unknown_command PROPERTIES WILL_FAIL TRUE)

# Training commands on a tiny hand-written corpus (fixtures/cli/README.md).
set(CLI_FIXTURES ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/cli)
set(CLI_OUT ${CMAKE_CURRENT_BINARY_DIR}/cli)

add_test(NAME cli_tokenizer_train
  COMMAND $<TARGET_FILE:titan> tokenizer train --type char
    --corpus ${CLI_FIXTURES}/tiny_corpus.txt --out ${CLI_OUT}/tok/tokenizer.json)

add_test(NAME cli_train_smoke
  COMMAND $<TARGET_FILE:titan> train --config ${CLI_FIXTURES}/tiny-config.json
    --data ${CLI_FIXTURES}/tiny_corpus.txt --out ${CLI_OUT}/smoke --device cpu)
set_tests_properties(cli_train_smoke PROPERTIES
  PASS_REGULAR_EXPRESSION "final val_loss [0-9.]+")

add_test(NAME cli_train_no_newline_corpus
  COMMAND $<TARGET_FILE:titan> train --config ${CLI_FIXTURES}/tiny-config.json
    --data ${CLI_FIXTURES}/no_newline_corpus.txt --out ${CLI_OUT}/no_newline
    --device cpu)
set_tests_properties(cli_train_no_newline_corpus PROPERTIES
  PASS_REGULAR_EXPRESSION "final val_loss [0-9.]+")

# expect_failure.cmake requires exit code 1 and the pattern in the output.
function(add_cli_failure name pattern)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND}
      -DEXE=$<TARGET_FILE:titan>
      "-DARGS=${ARGN}"
      "-DPATTERN=${pattern}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_failure.cmake)
endfunction()

add_cli_failure(cli_train_gemm_needs_cuda "--gemm needs --device cuda"
  "train;--config;${CLI_FIXTURES}/tiny-config.json;--data;${CLI_FIXTURES}/tiny_corpus.txt;--out;${CLI_OUT}/gemm_cpu;--device;cpu;--gemm;tiled")
add_cli_failure(cli_train_bad_config "n_kv_heads"
  "train;--config;${CLI_FIXTURES}/bad-config.json;--data;${CLI_FIXTURES}/tiny_corpus.txt;--out;${CLI_OUT}/bad;--device;cpu")
add_cli_failure(cli_train_short_corpus "too few for a training split"
  "train;--config;${CLI_FIXTURES}/tiny-config.json;--data;${CLI_FIXTURES}/short_corpus.txt;--out;${CLI_OUT}/short;--device;cpu")

# Usage errors exit with code 2 and print the usage text.
function(add_cli_usage name)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND}
      -DEXE=$<TARGET_FILE:titan>
      "-DARGS=${ARGN}"
      -DEXPECT_CODE=2
      "-DPATTERN=usage: titan"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_failure.cmake)
endfunction()

set(CLI_TRAIN_ARGS "train;--config;${CLI_FIXTURES}/tiny-config.json;--data;${CLI_FIXTURES}/tiny_corpus.txt;--out;${CLI_OUT}/usage")
add_cli_usage(cli_usage_bad_device "${CLI_TRAIN_ARGS};--device;gpu")
add_cli_usage(cli_usage_bad_gemm "${CLI_TRAIN_ARGS};--device;cpu;--gemm;fast")
add_cli_usage(cli_usage_bench_bare "bench")
add_cli_usage(cli_usage_bench_unknown "bench;conv")
add_cli_usage(cli_usage_sizes_zero "bench;matmul;--sizes;0")
add_cli_usage(cli_usage_sizes_empty_segment "bench;matmul;--sizes;256,,512")
add_cli_usage(cli_usage_sizes_not_a_number "bench;matmul;--sizes;abc")
add_cli_usage(cli_usage_sizes_trailing_comma "bench;matmul;--sizes;256,")
add_cli_usage(cli_usage_repeat_zero "bench;matmul;--repeat;0")

# With every CUDA device hidden, --device cuda must fail before writing --out.
add_test(NAME cli_train_cuda_no_gpu
  COMMAND ${CMAKE_COMMAND}
    -DEXE=$<TARGET_FILE:titan>
    -DCONFIG=${CLI_FIXTURES}/tiny-config.json
    -DDATA=${CLI_FIXTURES}/tiny_corpus.txt
    -DOUT_DIR=${CLI_OUT}/no_gpu
    -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_no_gpu_failure.cmake)

# CUDA CLI tests. They skip when the GPU is missing, which the commands report
# as "no CUDA device available".
add_test(NAME cli_train_cuda_smoke
  COMMAND $<TARGET_FILE:titan> train --config ${CLI_FIXTURES}/tiny-config.json
    --data ${CLI_FIXTURES}/tiny_corpus.txt --out ${CLI_OUT}/cuda_smoke
    --device cuda)
set_tests_properties(cli_train_cuda_smoke PROPERTIES
  PASS_REGULAR_EXPRESSION "final val_loss [0-9.]+"
  SKIP_REGULAR_EXPRESSION "no CUDA device available")

add_test(NAME cli_bench_matmul_smoke
  COMMAND $<TARGET_FILE:titan> bench matmul --sizes 64 --repeat 1)
set_tests_properties(cli_bench_matmul_smoke PROPERTIES
  PASS_REGULAR_EXPRESSION "tiled"
  SKIP_REGULAR_EXPRESSION "no CUDA device available")

# Checkpoints, resume and generation (M3). cli_train_writes_checkpoint leaves
# the checkpoint that the generate tests read.
add_test(NAME cli_train_writes_checkpoint
  COMMAND ${CMAKE_COMMAND}
    -DEXE=$<TARGET_FILE:titan>
    -DCONFIG=${CLI_FIXTURES}/checkpoint-config.json
    -DDATA=${CLI_FIXTURES}/tiny_corpus.txt
    -DOUT_DIR=${CLI_OUT}/ckpt
    -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_checkpoint.cmake)
set_tests_properties(cli_train_writes_checkpoint PROPERTIES
  FIXTURES_SETUP cli_checkpoint)

add_test(NAME cli_train_resume
  COMMAND ${CMAKE_COMMAND}
    -DEXE=$<TARGET_FILE:titan>
    -DCONFIG=${CLI_FIXTURES}/checkpoint-config.json
    -DDATA=${CLI_FIXTURES}/tiny_corpus.txt
    -DSTEPS=6
    -DSTOP_AT=3
    -DOUT_DIR=${CLI_OUT}/resume
    -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_train_resume.cmake)

add_cli_usage(cli_train_resume_rejects_config
  "train;--resume;${CLI_OUT}/ckpt;--config;${CLI_FIXTURES}/tiny-config.json")
add_cli_usage(cli_usage_stop_at_zero "${CLI_TRAIN_ARGS};--device;cpu;--stop-at;0")
add_cli_failure(cli_train_stop_at_beyond_steps "--stop-at 21 must be after"
  "train;--config;${CLI_FIXTURES}/tiny-config.json;--data;${CLI_FIXTURES}/tiny_corpus.txt;--out;${CLI_OUT}/stop_beyond;--device;cpu;--stop-at;21")
add_cli_failure(cli_train_resume_finished_stop_at
  "--stop-at 20 must be after the last completed step 20"
  "train;--resume;${CLI_OUT}/ckpt;--device;cpu;--stop-at;20")
set_tests_properties(cli_train_resume_finished_stop_at PROPERTIES
  FIXTURES_REQUIRED cli_checkpoint)

set(CLI_CKPT ${CLI_OUT}/ckpt/latest.ckpt)
set(CLI_PROMPT "First Citizen:")

add_test(NAME cli_generate_smoke
  COMMAND ${CMAKE_COMMAND}
    -DEXE=$<TARGET_FILE:titan>
    -DCHECKPOINT=${CLI_CKPT}
    "-DPROMPT=${CLI_PROMPT}"
    -DMAX_NEW=20
    "-DARGS=--device;cpu"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_generate.cmake)

add_test(NAME cli_generate_greedy_cache_equals_nocache
  COMMAND ${CMAKE_COMMAND}
    -DEXE=$<TARGET_FILE:titan>
    -DCHECKPOINT=${CLI_CKPT}
    "-DPROMPT=${CLI_PROMPT}"
    -DMAX_NEW=40
    "-DARGS=--device;cpu;--temp;0"
    "-DCOMPARE_ARGS=--device;cpu;--temp;0;--no-cache"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_generate.cmake)

add_test(NAME cli_generate_cuda_smoke
  COMMAND ${CMAKE_COMMAND}
    -DEXE=$<TARGET_FILE:titan>
    -DCHECKPOINT=${CLI_CKPT}
    "-DPROMPT=${CLI_PROMPT}"
    -DMAX_NEW=20
    "-DARGS=--device;cuda"
    -DCUDA_SKIP=ON
    -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_generate.cmake)
set_tests_properties(cli_generate_cuda_smoke PROPERTIES
  SKIP_REGULAR_EXPRESSION "no CUDA device available")

# Runtime failures of generate exit 1 with "error:" and print nothing to
# stdout.
function(add_cli_generate_failure name pattern)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND}
      -DEXE=$<TARGET_FILE:titan>
      "-DARGS=${ARGN}"
      "-DPATTERN=${pattern}"
      -DNO_STDOUT=ON
      -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_failure.cmake)
endfunction()

add_cli_generate_failure(cli_generate_unknown_char "error: .*U[+]005A"
  "generate;--checkpoint;${CLI_CKPT};--prompt;Zounds;--device;cpu")
add_cli_generate_failure(cli_generate_missing_checkpoint
  "error: .*missing[.]ckpt"
  "generate;--checkpoint;${CLI_OUT}/missing.ckpt;--device;cpu")

# Every bad sampling flag is a usage error, found before the (here missing)
# checkpoint is read.
set(CLI_GEN_ARGS "generate;--checkpoint;${CLI_OUT}/missing.ckpt;--device;cpu")
add_test(NAME cli_generate_bad_flags
  COMMAND ${CMAKE_COMMAND}
    -DEXE=$<TARGET_FILE:titan>
    "-DCASES=${CLI_GEN_ARGS};--top-p;0;|;${CLI_GEN_ARGS};--top-p;1.5;|;${CLI_GEN_ARGS};--top-k;-1;|;${CLI_GEN_ARGS};--temp;-1;|;${CLI_GEN_ARGS};--max-new;0;|;${CLI_GEN_ARGS};--no-cache;x;|;${CLI_GEN_ARGS};--seed;-3"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/harness/expect_usage_cases.cmake)

set_tests_properties(cli_generate_smoke cli_generate_greedy_cache_equals_nocache
  cli_generate_cuda_smoke cli_generate_unknown_char
  PROPERTIES FIXTURES_REQUIRED cli_checkpoint)
