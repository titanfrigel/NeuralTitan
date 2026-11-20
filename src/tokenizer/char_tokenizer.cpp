#include "tokenizer/char_tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>

#include "core/error.hpp"
#include "core/utf8.hpp"

namespace titan::tokenizer {

namespace {

// "U+00E9": uppercase hex, at least four digits.
std::string code_point_label(char32_t cp) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "U+%04X", static_cast<unsigned>(cp));
    return buf;
}

}  // namespace

CharTokenizer::CharTokenizer(std::vector<char32_t> vocab)
    : vocab_(std::move(vocab)) {
    for (std::size_t i = 0; i < vocab_.size(); ++i) {
        ids_.emplace(vocab_[i], static_cast<std::int32_t>(i));
    }
}

CharTokenizer CharTokenizer::build(std::string_view corpus) {
    std::vector<char32_t> vocab = decode_utf8(corpus);
    std::sort(vocab.begin(), vocab.end());
    vocab.erase(std::unique(vocab.begin(), vocab.end()), vocab.end());
    TITAN_CHECK(!vocab.empty(), "CharTokenizer::build: the corpus is empty");
    return CharTokenizer(std::move(vocab));
}

std::vector<std::int32_t> CharTokenizer::encode(std::string_view text) const {
    std::vector<std::int32_t> out;
    out.reserve(text.size());
    std::size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = 0;
        std::size_t length = 0;
        TITAN_CHECK(decode_utf8_at(text, pos, cp, length),
                    "CharTokenizer::encode: invalid UTF-8 at byte offset " +
                        std::to_string(pos));
        const auto it = ids_.find(cp);
        TITAN_CHECK(it != ids_.end(),
                    "CharTokenizer::encode: unknown character " +
                        code_point_label(cp) + " at byte offset " +
                        std::to_string(pos));
        out.push_back(it->second);
        pos += length;
    }
    return out;
}

std::string CharTokenizer::decode(std::span<const std::int32_t> ids) const {
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        TITAN_CHECK(
            ids[i] >= 0 && static_cast<std::size_t>(ids[i]) < vocab_.size(),
            "CharTokenizer::decode: id " + std::to_string(ids[i]) +
                " at position " + std::to_string(i) +
                " is outside the vocabulary of size " +
                std::to_string(vocab_.size()));
        out += encode_utf8(vocab_[static_cast<std::size_t>(ids[i])]);
    }
    return out;
}

std::int64_t CharTokenizer::vocab_size() const {
    return static_cast<std::int64_t>(vocab_.size());
}

io::JsonValue CharTokenizer::to_json() const {
    io::JsonValue vocab = io::JsonValue::array();
    for (const char32_t cp : vocab_) {
        vocab.push_back(io::JsonValue::string(encode_utf8(cp)));
    }
    io::JsonValue doc = io::JsonValue::object();
    doc.set("type", io::JsonValue::string("char"));
    doc.set("vocab", std::move(vocab));
    return doc;
}

CharTokenizer CharTokenizer::from_json(const io::JsonValue& json) {
    TITAN_CHECK(json.is_object(),
                "CharTokenizer::from_json: the document must be an object");
    const io::JsonValue& type = json.at("type");
    TITAN_CHECK(type.is_string(),
                "CharTokenizer::from_json: \"type\" must be a string");
    TITAN_CHECK(type.as_string() == "char",
                "CharTokenizer::from_json: unsupported type \"" +
                    type.as_string() + "\", expected \"char\"");
    const io::JsonValue& vocab_json = json.at("vocab");
    TITAN_CHECK(vocab_json.is_array(),
                "CharTokenizer::from_json: \"vocab\" must be an array");
    const io::JsonValue::Array& entries = vocab_json.as_array();
    TITAN_CHECK(!entries.empty(),
                "CharTokenizer::from_json: \"vocab\" is empty");

    std::vector<char32_t> vocab;
    vocab.reserve(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        TITAN_CHECK(entries[i].is_string(),
                    "CharTokenizer::from_json: vocab entry " +
                        std::to_string(i) + " must be a string");
        const std::vector<char32_t> cps = decode_utf8(entries[i].as_string());
        TITAN_CHECK(cps.size() == 1,
                    "CharTokenizer::from_json: vocab entry " +
                        std::to_string(i) +
                        " must be exactly one code point, found " +
                        std::to_string(cps.size()));
        TITAN_CHECK(
            std::find(vocab.begin(), vocab.end(), cps[0]) == vocab.end(),
            "CharTokenizer::from_json: vocab entry " + std::to_string(i) +
                " duplicates " + code_point_label(cps[0]));
        vocab.push_back(cps[0]);
    }
    return CharTokenizer(std::move(vocab));
}

}  // namespace titan::tokenizer
