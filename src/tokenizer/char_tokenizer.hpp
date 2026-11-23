#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "io/json.hpp"

namespace titan::tokenizer {

// Character-level tokenizer: one token per Unicode code point. The vocabulary
// is the sorted set of distinct code points of a corpus and the id of a
// character is its index in that list. Text and files are UTF-8; invalid
// UTF-8 and unknown characters raise titan::Error.
class CharTokenizer {
public:
    // The vocabulary is the distinct code points of `corpus`, ascending.
    static CharTokenizer build(std::string_view corpus);

    // Throws on invalid UTF-8 or on a character outside the vocabulary,
    // naming the code point (U+XXXX) and its byte offset.
    std::vector<std::int32_t> encode(std::string_view text) const;

    // Throws on an id outside [0, vocab_size()).
    std::string decode(std::span<const std::int32_t> ids) const;

    std::int64_t vocab_size() const;

    // {"type":"char","vocab":["\n"," ","!",...]}
    io::JsonValue to_json() const;
    // Validates the document: type, non-empty vocab, one code point per
    // entry, no duplicates.
    static CharTokenizer from_json(const io::JsonValue& json);

private:
    explicit CharTokenizer(std::vector<char32_t> vocab);

    std::vector<char32_t> vocab_;
    std::unordered_map<char32_t, std::int32_t> ids_;
};

}  // namespace titan::tokenizer
