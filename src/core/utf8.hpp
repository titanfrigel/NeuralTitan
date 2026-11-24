#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace titan {

// Decodes the UTF-8 sequence starting at text[pos]. On success stores the code
// point and the sequence length in bytes and returns true. Returns false for
// an invalid start: overlong forms, surrogates (U+D800..U+DFFF), values above
// U+10FFFF, a stray continuation byte or a truncated sequence. `pos` must be
// below text.size().
bool decode_utf8_at(std::string_view text, std::size_t pos,
                    char32_t& code_point, std::size_t& length);

// Decodes all of `text`. Throws titan::Error with the byte offset of the first
// invalid sequence.
std::vector<char32_t> decode_utf8(std::string_view text);

// Returns the UTF-8 encoding of `code_point`. Throws
// titan::Error for surrogates and values above U+10FFFF.
std::string encode_utf8(char32_t code_point);

// True when `text` is entirely valid UTF-8.
bool is_valid_utf8(std::string_view text);

// The length of `bytes` without a trailing incomplete UTF-8 sequence: a lead
// byte followed by fewer continuation bytes than it needs (at most 3 bytes in
// all) that some valid sequence starts with. Streaming output prints this
// prefix and keeps the rest until the next bytes arrive. Invalid bytes, and
// trailing bytes no valid sequence starts with (an overlong lead, the start
// of a surrogate or of a value above U+10FFFF), count as complete, so they
// are never held back.
std::size_t utf8_complete_prefix(std::string_view bytes);

// Streams bytes that may stop inside a UTF-8 character, such as the decoded
// tokens of a BPE tokenizer, so that no returned piece ends inside one.
class Utf8Stream {
public:
    // Appends `bytes` to the pending ones and returns their
    // utf8_complete_prefix, keeping the incomplete tail for a later push.
    std::string push(std::string_view bytes);
    // Returns what is still pending, as it is (an incomplete character when
    // the input stopped inside one), and clears it.
    std::string finish();

private:
    std::string pending_;
};

}  // namespace titan
