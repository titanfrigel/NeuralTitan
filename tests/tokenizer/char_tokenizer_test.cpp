#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "harness/test.hpp"
#include "io/json.hpp"
#include "tokenizer/char_tokenizer.hpp"

namespace {

using titan::io::JsonValue;
using titan::io::parse_json;
using titan::tokenizer::CharTokenizer;

std::string from_json_error(const std::string& text) {
    try {
        CharTokenizer::from_json(parse_json(text));
    } catch (const titan::Error& e) {
        return e.what();
    }
    return "";
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

TITAN_TEST(CharTokenizer, BuildSortedVocab) {
    const CharTokenizer tok = CharTokenizer::build("hello");
    TITAN_EXPECT_EQ(tok.vocab_size(), 4);
    // Vocabulary is e h l o, so "hello" is h=1 e=0 l=2 l=2 o=3.
    const std::vector<std::int32_t> expected = {1, 0, 2, 2, 3};
    TITAN_EXPECT(tok.encode("hello") == expected);
    TITAN_EXPECT_EQ(tok.decode(tok.encode("eho")), std::string("eho"));
}

TITAN_TEST(CharTokenizer, EncodeDecodeRoundTrip) {
    const std::string text = "caf\xC3\xA9\nna\xC3\xAFve \xE2\x82\xAC";
    const CharTokenizer tok = CharTokenizer::build(text);
    TITAN_EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TITAN_TEST(CharTokenizer, UnknownCharThrows) {
    const CharTokenizer tok = CharTokenizer::build("abc");
    std::string msg;
    try {
        tok.encode("ab\xC3\xA9");
    } catch (const titan::Error& e) {
        msg = e.what();
    }
    TITAN_EXPECT(contains(msg, "unknown character U+00E9"));
    TITAN_EXPECT(contains(msg, "byte offset 2"));
}

TITAN_TEST(CharTokenizer, DecodeOutOfRangeThrows) {
    const CharTokenizer tok = CharTokenizer::build("abc");
    const std::vector<std::int32_t> high = {0, 3};
    const std::vector<std::int32_t> low = {-1};
    TITAN_EXPECT_THROWS(tok.decode(high), titan::Error);
    TITAN_EXPECT_THROWS(tok.decode(low), titan::Error);
}

TITAN_TEST(CharTokenizer, InvalidUtf8Throws) {
    const CharTokenizer tok = CharTokenizer::build("abc");
    TITAN_EXPECT_THROWS(CharTokenizer::build("\xC3"), titan::Error);
    TITAN_EXPECT_THROWS(CharTokenizer::build("\xED\xA0\x80"), titan::Error);
    TITAN_EXPECT_THROWS(tok.encode("a\xC3"), titan::Error);
    TITAN_EXPECT_THROWS(tok.encode("\xED\xA0\x80"), titan::Error);
    TITAN_EXPECT_THROWS(CharTokenizer::build(""), titan::Error);
}

TITAN_TEST(CharTokenizer, JsonRoundTrip) {
    const std::string text = "caf\xC3\xA9\n \"quoted\"";
    const CharTokenizer tok = CharTokenizer::build(text);
    const std::string dumped = titan::io::to_json(tok.to_json());
    const CharTokenizer loaded = CharTokenizer::from_json(parse_json(dumped));
    TITAN_EXPECT_EQ(loaded.vocab_size(), tok.vocab_size());
    TITAN_EXPECT(loaded.encode(text) == tok.encode(text));
    TITAN_EXPECT_EQ(loaded.decode(loaded.encode(text)), text);
    const JsonValue doc = tok.to_json();
    TITAN_EXPECT_EQ(doc.at("type").as_string(), std::string("char"));
}

TITAN_TEST(CharTokenizer, RejectsMalformedJson) {
    TITAN_EXPECT(contains(from_json_error("[]"), "must be an object"));
    TITAN_EXPECT(contains(from_json_error(R"({"vocab":["a"]})"), "type"));
    TITAN_EXPECT(contains(from_json_error(R"({"type":3,"vocab":["a"]})"),
                          "must be a string"));
    TITAN_EXPECT(contains(from_json_error(R"({"type":"bpe","vocab":["a"]})"),
                          "unsupported type"));
    TITAN_EXPECT(contains(from_json_error(R"({"type":"char"})"), "vocab"));
    TITAN_EXPECT(contains(from_json_error(R"({"type":"char","vocab":"a"})"),
                          "must be an array"));
    TITAN_EXPECT(
        contains(from_json_error(R"({"type":"char","vocab":[]})"), "is empty"));
    TITAN_EXPECT(contains(from_json_error(R"({"type":"char","vocab":["a",1]})"),
                          "entry 1 must be a string"));
    TITAN_EXPECT(
        contains(from_json_error(R"({"type":"char","vocab":["a",""]})"),
                 "entry 1 must be exactly one code point"));
    TITAN_EXPECT(contains(from_json_error(R"({"type":"char","vocab":["ab"]})"),
                          "entry 0 must be exactly one code point"));
    TITAN_EXPECT(
        contains(from_json_error(R"({"type":"char","vocab":["a","b","a"]})"),
                 "entry 2 duplicates U+0061"));
}
