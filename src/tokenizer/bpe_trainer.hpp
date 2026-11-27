#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "tokenizer/bpe_tokenizer.hpp"

namespace titan::tokenizer {

// One learned merge: the ids of its two parts and the number of times the
// pair occurred, weighted by word frequency, when it was chosen. Ids follow
// the trained vocabulary: 0 is <|endoftext|>, byte b is 1 + b, and merge i
// creates id 257 + i.
struct BpeMerge {
    std::int32_t left;
    std::int32_t right;
    std::int64_t count;
};

// Learns up to `max_merges` byte-level BPE merges from `corpus`
// (docs/math/bpe.md, "Training"):
//   - the corpus is split at every <|endoftext|>, which is removed, and each
//     span is pre-tokenized; the distinct pieces are the words;
//   - each step merges the pair with the highest count over the words,
//     weighted by word frequency; ties go to the smallest (left, right);
//   - every merge creates a new token (an internal invariant, checked);
//   - training stops when the best count is below 2.
// Keeps the pair counts and, per pair, the words that contain it, and
// updates them only for the words a merge changes. Throws titan::Error on
// invalid UTF-8 (naming the byte offset) or a negative `max_merges`.
std::vector<BpeMerge> bpe_merges(std::string_view corpus,
                                 std::int64_t max_merges);

// The same merges, recounting every pair of every word at each step. It is
// the reference the incremental version is tested against.
std::vector<BpeMerge> bpe_merges_naive(std::string_view corpus,
                                       std::int64_t max_merges);

// A tokenizer with <|endoftext|> at id 0, the 256 byte symbols at ids 1-256
// and up to vocab_size - 257 merges from bpe_merges. Its vocab_size() is
// 257 + the number of merges, which is smaller than `vocab_size` when the
// corpus runs out of repeating pairs. Throws titan::Error naming
// `vocab_size` when it is outside [257, 2^31).
BpeTokenizer train_bpe(std::string_view corpus, std::int64_t vocab_size);

}  // namespace titan::tokenizer
