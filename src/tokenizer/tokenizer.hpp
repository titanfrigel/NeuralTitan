#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "io/json.hpp"

namespace titan::tokenizer {

// What training, checkpoints and generation need from a tokenizer, so they
// work with the character-level and the BPE tokenizer alike.
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<std::int32_t> encode(std::string_view text) const = 0;
    // The bytes of `ids`; a BPE decode may end in the middle of a UTF-8
    // sequence (see utf8_complete_prefix).
    virtual std::string decode(std::span<const std::int32_t> ids) const = 0;
    virtual std::int64_t vocab_size() const = 0;
    // The end-of-text token, when the tokenizer has one.
    virtual std::optional<std::int32_t> eos_id() const = 0;
    // The document load_tokenizer reads back.
    virtual io::JsonValue to_json() const = 0;

protected:
    Tokenizer() = default;
    Tokenizer(const Tokenizer&) = default;
    Tokenizer(Tokenizer&&) = default;
    Tokenizer& operator=(const Tokenizer&) = default;
    Tokenizer& operator=(Tokenizer&&) = default;
};

// Builds the tokenizer a JSON document describes:
//   - an object with "type": "char" gives a CharTokenizer;
//   - an object whose "model" is an object (a Hugging Face tokenizer.json)
//     gives a BpeTokenizer.
// Anything else throws titan::Error "unsupported tokenizer document" naming
// what was found. Errors of the concrete parsers propagate unchanged.
std::unique_ptr<Tokenizer> load_tokenizer(const io::JsonValue& json);

}  // namespace titan::tokenizer
