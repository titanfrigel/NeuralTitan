#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "tokenizer/tokenizer.hpp"

namespace titan::tokenizer {

// The ids of `text`, equal to tok.encode(text), computed on num_threads()
// threads (docs/math/bpe.md, "Encoding a corpus in chunks").
//
// A tokenizer with an end-of-text token (eos_id) has the text cut right after
// each "<|endoftext|>"; the chunks are grouped into one range of roughly equal
// bytes per thread and each range is encoded on its own thread. Without one
// the text is encoded whole. An encode error propagates as encode throws it,
// except that a byte offset it names is relative to the start of its range.
std::vector<std::int32_t> encode_chunked(const Tokenizer& tok,
                                         std::string_view text);

}  // namespace titan::tokenizer
