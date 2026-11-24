// load_tokenizer: the document's shape picks the concrete tokenizer.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "core/error.hpp"
#include "harness/test.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "tokenizer/bpe_tokenizer.hpp"
#include "tokenizer/char_tokenizer.hpp"
#include "tokenizer/tokenizer.hpp"

using titan::io::parse_json;
using titan::tokenizer::BpeTokenizer;
using titan::tokenizer::CharTokenizer;
using titan::tokenizer::load_tokenizer;
using titan::tokenizer::Tokenizer;

namespace {

// The message of the titan::Error load_tokenizer throws for `text`, or "".
std::string load_error(const char* text) {
    try {
        load_tokenizer(parse_json(text));
    } catch (const titan::Error& e) {
        return e.what();
    }
    return "";
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

}  // namespace

TITAN_TEST(Tokenizer, LoadsBoth) {
    const titan::io::JsonValue char_doc =
        parse_json(R"({"type": "char", "vocab": ["\n", "a", "b"]})");
    const std::unique_ptr<Tokenizer> chars = load_tokenizer(char_doc);
    const CharTokenizer char_want = CharTokenizer::from_json(char_doc);
    TITAN_EXPECT(dynamic_cast<const CharTokenizer*>(chars.get()) != nullptr);
    TITAN_EXPECT_EQ(chars->vocab_size(), char_want.vocab_size());
    TITAN_EXPECT(chars->eos_id() == char_want.eos_id());
    TITAN_EXPECT(!chars->eos_id().has_value());

    const titan::io::JsonValue bpe_doc = parse_json(
        titan::io::read_file(std::filesystem::path(TITAN_FIXTURES_DIR) /
                             "smollm2" / "tokenizer.json"));
    const std::unique_ptr<Tokenizer> bpe = load_tokenizer(bpe_doc);
    const BpeTokenizer bpe_want = BpeTokenizer::from_json(bpe_doc);
    TITAN_EXPECT(dynamic_cast<const BpeTokenizer*>(bpe.get()) != nullptr);
    TITAN_EXPECT_EQ(bpe->vocab_size(), bpe_want.vocab_size());
    TITAN_EXPECT(bpe->eos_id() == bpe_want.eos_id());
    TITAN_EXPECT(bpe->eos_id().has_value());
}

TITAN_TEST(Tokenizer, RejectsUnknownDocument) {
    const std::string empty = load_error("{}");
    TITAN_EXPECT(contains(empty, "unsupported tokenizer document"));
    TITAN_EXPECT(contains(empty, R"(no "type" or "model" key)"));
    const std::string word = load_error(R"({"type": "word"})");
    TITAN_EXPECT(contains(word, "unsupported tokenizer document"));
    TITAN_EXPECT(contains(word, R"(type "word")"));
    TITAN_EXPECT(contains(load_error("[]"), "an array"));
    TITAN_EXPECT(contains(load_error(R"({"model": 3})"),
                          "unsupported tokenizer document"));
    // Errors of the concrete parsers pass through unchanged.
    TITAN_EXPECT(
        contains(load_error(R"({"type": "char", "vocab": []})"), "vocab"));
}
