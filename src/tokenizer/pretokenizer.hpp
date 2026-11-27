#pragma once

#include <string_view>
#include <vector>

namespace titan::tokenizer {

// Splits UTF-8 `text` into the pieces that byte-level BPE encodes one at a
// time, as SmolLM2's pre-tokenizer does (docs/math/bpe.md, "Pre-tokenizer"):
//   1. every number of the Digits stage (is_numeric) is a piece of its own;
//   2. each remaining piece is split by the GPT-2 pattern:
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// The pieces are views into `text`, in order, non-empty, and concatenate to
// `text`. Throws titan::Error naming the byte offset of invalid UTF-8.
std::vector<std::string_view> pretokenize(std::string_view text);

}  // namespace titan::tokenizer
