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
//  [--tokenizer <tok.json>] [--steps N] [--device cpu|cuda]
//  [--gemm cublas|naive|tiled]`. The device defaults to cuda when a GPU is
// available, otherwise cpu. `args` are the arguments after "train".
void train(std::span<const std::string_view> args);

// `titan bench matmul [--sizes 256,512,1024,2048,4096] [--repeat 10]`: times
// square GEMMs with every CUDA GEMM backend and prints them against the GPU's
// peak FP32 throughput. `args` are the arguments after "bench".
void bench(std::span<const std::string_view> args);

}  // namespace titan::cli
