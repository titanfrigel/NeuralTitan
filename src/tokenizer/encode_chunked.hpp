#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "tokenizer/tokenizer.hpp"

namespace titan::tokenizer {

// Splits [0, text.size()) into byte ranges [begin, end) that start and end
// right after an "<|endoftext|>" (or at the ends of the text). A range grows
// chunk by chunk until it has text.size() / parts bytes, so a long
// marker-free chunk makes a longer range and there may be fewer than `parts`
// ranges. Exposed for tests; parts >= 1.
std::vector<std::pair<std::size_t, std::size_t>> split_at_end_of_text(
    std::string_view text, std::size_t parts);

// The ids of `text`, equal to tok.encode(text), computed on num_threads()
// threads (docs/math/bpe.md, "Encoding a corpus in chunks").
//
// A tokenizer with an end-of-text token (eos_id) has the text cut right after
// each "<|endoftext|>"; the chunks are grouped into 4 ranges of roughly equal
// bytes per thread (split_at_end_of_text) and the ranges are encoded through
// parallel_for, each thread taking a block of them. Without one
// the text is encoded whole. An encode error propagates as encode throws it,
// except that a byte offset it names is relative to the start of its range.
std::vector<std::int32_t> encode_chunked(const Tokenizer& tok,
                                         std::string_view text);

}  // namespace titan::tokenizer
