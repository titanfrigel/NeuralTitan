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
//  [--tokenizer <tok.json>] [--steps N] [--device cpu|cuda]`.
// `args` are the arguments after "train".
void train(std::span<const std::string_view> args);

}  // namespace titan::cli
