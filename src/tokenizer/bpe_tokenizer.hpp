#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "io/json.hpp"

namespace titan::tokenizer {

// Byte-level BPE tokenizer, as SmolLM2-135M uses it (docs/math/bpe.md).
// Added (special) tokens such as <|endoftext|> are split off first; the rest
// of the text is pre-tokenized, mapped to the byte-level alphabet and merged
// by rank. It reads and writes the Hugging Face `tokenizer.json` layout,
// restricted to the features SmolLM2 uses.
class BpeTokenizer {
public:
    // Validates a Hugging Face `tokenizer.json` document. Every unsupported or
    // malformed field throws titan::Error naming the field path, such as
    // "model.merges[3]" or "added_tokens[0].id".
    static BpeTokenizer from_json(const io::JsonValue& json);

    // Builds a tokenizer from in-memory parts (the trainer's output).
    //   - vocab[id] is the token's text in the byte-level alphabet, or the
    //     literal text of an added token;
    //   - merges[rank] = (left id, right id); the merged token is the
    //     vocabulary entry vocab[left] + vocab[right];
    //   - a part of a merge is a single byte symbol or the result of an
    //     earlier merge, and no two merges create the same token;
    //   - added_ids are the ids of the added tokens, none of which is a
    //     single byte symbol.
    // A byte may lack a symbol; encode then drops it.
    // Throws titan::Error on parts that break these rules.
    static BpeTokenizer from_parts(
        std::vector<std::string> vocab,
        std::vector<std::pair<std::int32_t, std::int32_t>> merges,
        std::vector<std::int32_t> added_ids);

    // The `tokenizer.json` layout with "version": "1.0", SmolLM2's
    // pre-tokenizer and decoder, and merges written as "a b" strings.
    io::JsonValue to_json() const;

    // A byte whose symbol is not in the vocabulary yields no id, as in the
    // reference (unk_token is null). Throws titan::Error naming the byte
    // offset of invalid UTF-8.
    std::vector<std::int32_t> encode(std::string_view text) const;

    // Raw bytes, which may end in the middle of a UTF-8 sequence. An added
    // token yields its literal text. Throws on an id outside
    // [0, vocab_size()), naming the id and its position.
    std::string decode(std::span<const std::int32_t> ids) const;

    std::int64_t vocab_size() const;
    // The id of the added token <|endoftext|>, if there is one.
    std::optional<std::int32_t> eos_id() const;
    std::int64_t num_merges() const;

private:
    // The token a merge produces and the merge's rank (lower merges first).
    struct Merge {
        std::int32_t rank;
        std::int32_t merged;
    };

    // byte_ids_ entry of a byte whose symbol is not an ordinary token.
    static constexpr std::int32_t kNoSymbol = -1;

    BpeTokenizer() = default;

    static std::uint64_t pair_key(std::int32_t left, std::int32_t right);

    // Appends the ids of one pre-tokenized piece.
    void encode_piece(std::string_view piece,
                      std::vector<std::int32_t>& out) const;

    std::vector<std::string> vocab_;   // id -> text (byte-level or literal)
    std::vector<std::string> bytes_;   // id -> the bytes it decodes to
    std::vector<bool> is_added_;       // id -> is an added token
    std::vector<std::int32_t> added_;  // added ids, in the order given
    std::unordered_map<std::string, std::int32_t> ids_;  // text -> id
    std::array<std::int32_t, 256>
        byte_ids_{};  // byte -> symbol id or kNoSymbol
    std::vector<std::pair<std::int32_t, std::int32_t>> merges_;  // by rank
    std::unordered_map<std::uint64_t, Merge> merge_of_;  // (left, right)
};

}  // namespace titan::tokenizer
