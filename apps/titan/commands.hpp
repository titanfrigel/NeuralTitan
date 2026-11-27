#pragma once

#include <span>
#include <string_view>

namespace titan::cli {

// Thrown for a malformed command line; main prints the usage and exits 2.
struct UsageError {};

// `titan tokenizer train --type char --corpus <file> --out <tok.json>` builds
// a character tokenizer; `titan tokenizer train --type bpe --corpus <file>
// --vocab N --out <tok.json>` trains a byte-level BPE tokenizer of at most N
// tokens, N in [257, 262144], and prints its size, merges and time.
// `args` are the arguments after "tokenizer train".
void tokenizer_train(std::span<const std::string_view> args);

// `titan tokenizer encode --tokenizer <tok.json> (--text <s> | --file <path>)
//  [--summary]` encodes the text with a character or BPE tokenizer and prints
// the ids on one line, separated by spaces. With --summary it prints
// `tokens N`, `crc32 0x%08x` (the CRC-32 of the ids as little-endian int32,
// as in tests/fixtures/smollm2/corpus.json) and `bytes_per_token X.XXX`
// instead. `args` are the arguments after "tokenizer encode".
void tokenizer_encode(std::span<const std::string_view> args);

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

// `titan generate (--checkpoint <file> | --hf-model <dir>) [--prompt <text>]
//  [--max-new N] [--temp T] [--top-k K] [--top-p P] [--seed S]
//  [--device cpu|cuda] [--no-cache]`: echoes the prompt (default a newline),
// then streams the text of each sampled token to stdout, holding back the
// bytes of an incomplete UTF-8 character until a later token completes it,
// and writes tokens/s to stderr. Generation stops early at the tokenizer's
// end-of-text token, which is not printed. --hf-model loads a Hugging Face
// Llama directory (config.json, tokenizer.json, model.safetensors) with
// runtime::load_hf_model; exactly one of --checkpoint and --hf-model is
// required. Defaults: --temp 0.8 --top-k 0 --top-p 0.9 --max-new 500, --seed
// the checkpoint's train.seed or 0 with --hf-model, --device cuda when a GPU
// is available. --no-cache recomputes the whole window per token
// (generate_naive) for comparison. A file that fails to load prints nothing
// to stdout. `args` are the arguments after "generate".
void generate(std::span<const std::string_view> args);

// `titan bench matmul [--sizes 256,512,1024,2048,4096] [--repeat 10]`: times
// square GEMMs with every CUDA GEMM backend and prints them against the GPU's
// peak FP32 throughput. `args` are the arguments after "bench".
void bench(std::span<const std::string_view> args);

}  // namespace titan::cli
