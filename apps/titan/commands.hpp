#pragma once

#include <span>
#include <string_view>

namespace titan::cli {

// Thrown for a malformed command line; main prints the usage and exits 2.
struct UsageError {};

// `titan tokenizer train --type char --corpus <file> --out <tok.json>`.
// `args` are the arguments after "tokenizer train".
void tokenizer_train(std::span<const std::string_view> args);

// `titan train --config <cfg.json> --data <corpus.txt> --out <dir>
//  [--tokenizer <tok.json>] [--steps N] [--stop-at N] [--device cpu|cuda]
//  [--gemm cublas|naive|tiled]` starts a run;
// `titan train --resume <dir> [--stop-at N] [--device cpu|cuda]
//  [--gemm cublas|naive|tiled]` continues the run saved in <dir>/latest.ckpt.
// Checkpoints go to <dir>/latest.ckpt (train.checkpoint_interval, the last
// step, and --stop-at). --stop-at N stops after step N, which must lie after
// the last completed step and at most train.steps. The device defaults to cuda
// when a GPU is available, otherwise cpu. `args` are the arguments after
// "train".
void train(std::span<const std::string_view> args);

// `titan generate --checkpoint <file> [--prompt <text>] [--max-new N]
//  [--temp T] [--top-k K] [--top-p P] [--seed S] [--device cpu|cuda]
//  [--no-cache]`: echoes the prompt (default a newline), then streams each
// sampled character to stdout, and writes tokens/s to stderr. Defaults:
// --temp 0.8 --top-k 0 --top-p 0.9 --max-new 500, --seed the checkpoint's
// train.seed. --no-cache recomputes the whole window per token
// (generate_naive) for comparison. `args` are the arguments after
// "generate".
void generate(std::span<const std::string_view> args);

// `titan bench matmul [--sizes 256,512,1024,2048,4096] [--repeat 10]`: times
// square GEMMs with every CUDA GEMM backend and prints them against the GPU's
// peak FP32 throughput. `args` are the arguments after "bench".
void bench(std::span<const std::string_view> args);

}  // namespace titan::cli
