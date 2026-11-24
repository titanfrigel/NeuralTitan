#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "core/utf8.hpp"
#include "harness/test.hpp"
#include "io/json.hpp"
#include "support/random_text.hpp"
#include "tokenizer/bpe_tokenizer.hpp"
#include "tokenizer/byte_level.hpp"

namespace {

using titan::io::JsonValue;
using titan::io::parse_json;
using titan::tokenizer::BpeTokenizer;
using titan::tokenizer::byte_to_symbol;
using titan::tokenizer::bytes_to_symbols;
using titan::tokenizer::symbol_to_byte;
using titan::tokenizer::symbols_to_bytes;

constexpr std::string_view kPreTokenizer =
    R"({"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},)"
    R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]})";
constexpr std::string_view kDecoder =
    R"({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true})";

// The byte-level symbol of `byte` as UTF-8.
std::string sym(unsigned char byte) {
    return titan::encode_utf8(byte_to_symbol(byte));
}

// In the test tokenizer, <|endoftext|> is id 0 and byte b is id 1 + b.
std::int32_t byte_id(char byte) {
    return 1 + static_cast<std::int32_t>(static_cast<unsigned char>(byte));
}

JsonValue string_merges(std::initializer_list<std::string_view> merges) {
    JsonValue out = JsonValue::array();
    for (const std::string_view m : merges) {
        out.push_back(JsonValue::string(std::string(m)));
    }
    return out;
}

JsonValue array_merges(
    std::initializer_list<std::pair<std::string_view, std::string_view>>
        merges) {
    JsonValue out = JsonValue::array();
    for (const auto& [a, b] : merges) {
        JsonValue pair = JsonValue::array();
        pair.push_back(JsonValue::string(std::string(a)));
        pair.push_back(JsonValue::string(std::string(b)));
        out.push_back(std::move(pair));
    }
    return out;
}

// A tokenizer.json in SmolLM2's layout: <|endoftext|> = 0, the 256 byte
// symbols in byte order (ids 1-256), then `extra` from id 257. When
// `missing_byte` is a byte, its symbol is left out and later ids move down.
JsonValue make_doc(std::initializer_list<std::string_view> extra,
                   JsonValue merges, int missing_byte = -1) {
    JsonValue vocab = JsonValue::object();
    std::int32_t id = 0;
    vocab.set("<|endoftext|>", JsonValue::number(id++));
    for (unsigned b = 0; b < 256; ++b) {
        if (static_cast<int>(b) == missing_byte) {
            continue;
        }
        vocab.set(sym(static_cast<unsigned char>(b)), JsonValue::number(id++));
    }
    for (const std::string_view text : extra) {
        vocab.set(std::string(text), JsonValue::number(id++));
    }

    JsonValue model = JsonValue::object();
    model.set("type", JsonValue::string("BPE"));
    model.set("dropout", JsonValue());
    model.set("unk_token", JsonValue());
    model.set("continuing_subword_prefix", JsonValue());
    model.set("end_of_word_suffix", JsonValue());
    model.set("fuse_unk", JsonValue::boolean(false));
    model.set("byte_fallback", JsonValue::boolean(false));
    model.set("ignore_merges", JsonValue::boolean(false));
    model.set("vocab", std::move(vocab));
    model.set("merges", std::move(merges));

    JsonValue eos = JsonValue::object();
    eos.set("id", JsonValue::number(0));
    eos.set("content", JsonValue::string("<|endoftext|>"));
    eos.set("single_word", JsonValue::boolean(false));
    eos.set("lstrip", JsonValue::boolean(false));
    eos.set("rstrip", JsonValue::boolean(false));
    eos.set("normalized", JsonValue::boolean(false));
    eos.set("special", JsonValue::boolean(true));
    JsonValue added = JsonValue::array();
    added.push_back(std::move(eos));

    JsonValue doc = JsonValue::object();
    doc.set("version", JsonValue::string("1.0"));
    doc.set("truncation", JsonValue());
    doc.set("padding", JsonValue());
    doc.set("added_tokens", std::move(added));
    doc.set("normalizer", JsonValue());
    doc.set("pre_tokenizer", parse_json(kPreTokenizer));
    doc.set("post_processor", JsonValue());
    doc.set("decoder", parse_json(kDecoder));
    doc.set("model", std::move(model));
    return doc;
}

// The brief's tokenizer: he = 257, ll = 258, hell = 259, Ġw = 260.
JsonValue hello_doc() {
    return make_doc({"he", "ll", "hell", "\xC4\xA0w"},
                    string_merges({"h e", "l l", "he ll", "\xC4\xA0 w"}));
}

// hello_doc() as one compact line, the base of the rejection tests.
std::string hello_text() { return titan::io::to_json(hello_doc(), 0); }

BpeTokenizer hello_tokenizer() { return BpeTokenizer::from_json(hello_doc()); }

// `text` with the first `from` replaced by `to`; empty when `from` is absent,
// which fails the test with a parse error instead of passing silently.
std::string replaced(const std::string& text, std::string_view from,
                     std::string_view to) {
    const std::size_t pos = text.find(from);
    if (pos == std::string::npos) {
        return "";
    }
    std::string out = text;
    out.replace(pos, from.size(), to);
    return out;
}

// The message from_json throws for `text`, or "" when it accepts it.
std::string from_json_error(const std::string& text) {
    try {
        BpeTokenizer::from_json(parse_json(text));
    } catch (const titan::Error& e) {
        return e.what();
    }
    return "";
}

std::string from_json_error(const JsonValue& doc) {
    return from_json_error(titan::io::to_json(doc, 0));
}

bool contains(const std::string& haystack, std::string_view needle) {
    return haystack.find(needle) != std::string::npos;
}

std::vector<std::int32_t> ids(std::initializer_list<std::int32_t> list) {
    return std::vector<std::int32_t>(list);
}

}  // namespace

TITAN_TEST(ByteLevel, RoundTrip) {
    for (unsigned b = 0; b < 256; ++b) {
        const auto byte = static_cast<std::uint8_t>(b);
        const std::optional<std::uint8_t> back =
            symbol_to_byte(byte_to_symbol(byte));
        TITAN_EXPECT(back.has_value() && *back == byte);
    }
    TITAN_EXPECT_EQ(static_cast<std::uint32_t>(byte_to_symbol(' ')), 0x120u);
    TITAN_EXPECT_EQ(static_cast<std::uint32_t>(byte_to_symbol('\n')), 0x10Au);
    TITAN_EXPECT_EQ(static_cast<std::uint32_t>(byte_to_symbol('a')),
                    static_cast<std::uint32_t>('a'));
    // The last of the 68 shifted bytes is 0xAD (soft hyphen): U+0143.
    TITAN_EXPECT_EQ(static_cast<std::uint32_t>(byte_to_symbol(0xAD)), 0x143u);
    TITAN_EXPECT(!symbol_to_byte(U' ').has_value());
    TITAN_EXPECT(!symbol_to_byte(0x144).has_value());

    std::string all;
    for (unsigned b = 0; b < 256; ++b) {
        all += static_cast<char>(b);
    }
    const std::optional<std::string> back =
        symbols_to_bytes(bytes_to_symbols(all));
    TITAN_EXPECT(back.has_value() && *back == all);
    TITAN_EXPECT(!symbols_to_bytes("a b").has_value());
}

TITAN_TEST(Bpe, EncodesByRank) {
    const BpeTokenizer tok = hello_tokenizer();
    TITAN_EXPECT_EQ(tok.vocab_size(), 261);
    TITAN_EXPECT_EQ(tok.num_merges(), 4);
    // hello: h e l l o -> he l l o -> he ll o -> hell o.
    // " world": Ġ w o r l d -> Ġw o r l d.
    const std::vector<std::int32_t> expected = {
        259,          byte_id('o'), 260,         byte_id('o'),
        byte_id('r'), byte_id('l'), byte_id('d')};
    TITAN_EXPECT(tok.encode("hello world") == expected);

    // When two merges compete for the same symbol, the lower rank wins:
    // ab = 257, bc = 258.
    const BpeTokenizer ab_first = BpeTokenizer::from_json(
        make_doc({"ab", "bc"}, string_merges({"a b", "b c"})));
    TITAN_EXPECT(ab_first.encode("abc") == ids({257, byte_id('c')}));
    const BpeTokenizer bc_first = BpeTokenizer::from_json(
        make_doc({"ab", "bc"}, string_merges({"b c", "a b"})));
    TITAN_EXPECT(bc_first.encode("abc") == ids({byte_id('a'), 258}));
}

TITAN_TEST(Bpe, MergesLeftToRight) {
    const BpeTokenizer tok =
        BpeTokenizer::from_json(make_doc({"aa"}, string_merges({"a a"})));
    TITAN_EXPECT(tok.encode("aaa") == ids({257, byte_id('a')}));
    TITAN_EXPECT(tok.encode("aaaa") == ids({257, 257}));
}

TITAN_TEST(Bpe, AddedTokensSplitFirst) {
    const BpeTokenizer tok = hello_tokenizer();
    TITAN_EXPECT(tok.eos_id() == std::optional<std::int32_t>(0));
    TITAN_EXPECT(tok.encode("x<|endoftext|>y") ==
                 ids({byte_id('x'), 0, byte_id('y')}));
    TITAN_EXPECT(tok.encode("<|endoftext|><|endoftext|>") == ids({0, 0}));
    // A partial added token is ordinary text: "<|" then "endoftext".
    const std::vector<std::int32_t> partial = tok.encode("<|endoftext");
    TITAN_EXPECT_EQ(partial.size(), std::size_t{11});
    TITAN_EXPECT_EQ(partial.front(), byte_id('<'));
}

TITAN_TEST(Bpe, EncodeRejectsInvalidUtf8) {
    const BpeTokenizer tok = hello_tokenizer();
    std::string msg;
    try {
        tok.encode("ab<|endoftext|>\xC3");
    } catch (const titan::Error& e) {
        msg = e.what();
    }
    TITAN_EXPECT(contains(msg, "byte offset 15"));
}

TITAN_TEST(Bpe, DropsBytesWithoutSymbol) {
    // Byte 0x04 has no symbol, as in SmolLM2, so byte b > 4 is id b, and
    // he = 256, ll = 257, hell = 258, Ġw = 259, !! = 260.
    const BpeTokenizer tok = BpeTokenizer::from_json(make_doc(
        {"he", "ll", "hell", "\xC4\xA0w", "!!"},
        string_merges({"h e", "l l", "he ll", "\xC4\xA0 w", "! !"}), 0x04));
    TITAN_EXPECT_EQ(tok.vocab_size(), 261);
    TITAN_EXPECT(tok.encode("a\x04"
                            "b") == ids({'a', 'b'}));
    TITAN_EXPECT(tok.encode("\x04\x04").empty());
    // Inside one piece the byte is dropped before merging, so its
    // neighbours merge: SmolLM2 encodes "!\x04!" as "!!".
    TITAN_EXPECT(tok.encode("!\x04!") == ids({260}));
    TITAN_EXPECT(tok.encode("!\x04\x04!") == ids({260}));
    // 0x04 is an "other" character, so the pre-tokenizer cuts "he\x04llo"
    // into he, \x04, llo: no merge spans the cut (SmolLM2 gives he, llo).
    TITAN_EXPECT(tok.encode("he\x04llo") == ids({256, 257, 'o'}));
    TITAN_EXPECT(tok.encode("hello") == ids({258, 'o'}));
    // So decode cannot restore it.
    TITAN_EXPECT_EQ(tok.decode(tok.encode("a\x04"
                                          "b")),
                    std::string("ab"));
}

TITAN_TEST(Bpe, DecodeIsInverse) {
    const BpeTokenizer tok = hello_tokenizer();
    std::vector<std::string> texts = titan::test::random_texts(7, 200);
    texts.push_back("x<|endoftext|>");
    texts.push_back("hello world\n");
    for (const std::string& text : texts) {
        TITAN_EXPECT_EQ(tok.decode(tok.encode(text)), text);
    }
    // A single id may stop inside a UTF-8 sequence: raw bytes come back.
    const std::vector<std::int32_t> half = {byte_id('\xC3')};
    TITAN_EXPECT_EQ(tok.decode(half), std::string("\xC3"));
}

TITAN_TEST(Bpe, DecodeRejectsBadId) {
    const BpeTokenizer tok = hello_tokenizer();
    for (const std::int32_t bad : {std::int32_t{-1}, std::int32_t{261}}) {
        const std::vector<std::int32_t> input = {1, bad};
        std::string msg;
        try {
            tok.decode(input);
        } catch (const titan::Error& e) {
            msg = e.what();
        }
        TITAN_EXPECT(contains(msg, "id " + std::to_string(bad)));
    }
}

TITAN_TEST(Bpe, JsonRoundTrip) {
    const BpeTokenizer tok = hello_tokenizer();
    const std::string dumped = titan::io::to_json(tok.to_json());
    const BpeTokenizer loaded = BpeTokenizer::from_json(parse_json(dumped));
    TITAN_EXPECT_EQ(loaded.vocab_size(), tok.vocab_size());
    TITAN_EXPECT_EQ(loaded.num_merges(), tok.num_merges());
    TITAN_EXPECT(loaded.eos_id() == tok.eos_id());
    // The written document is the input layout, field for field.
    TITAN_EXPECT_EQ(titan::io::to_json(tok.to_json(), 0), hello_text());
    for (const std::string& text : titan::test::random_texts(7, 200)) {
        TITAN_EXPECT(loaded.encode(text) == tok.encode(text));
    }
}

TITAN_TEST(Bpe, AcceptsArrayMerges) {
    const BpeTokenizer strings = hello_tokenizer();
    const BpeTokenizer arrays = BpeTokenizer::from_json(make_doc(
        {"he", "ll", "hell", "\xC4\xA0w"},
        array_merges(
            {{"h", "e"}, {"l", "l"}, {"he", "ll"}, {"\xC4\xA0", "w"}})));
    for (const std::string& text : titan::test::random_texts(7, 200)) {
        TITAN_EXPECT(arrays.encode(text) == strings.encode(text));
    }
    TITAN_EXPECT(arrays.encode("hello world") == strings.encode("hello world"));
}

TITAN_TEST(Bpe, FromPartsMatchesJson) {
    std::vector<std::string> vocab = {"<|endoftext|>"};
    for (unsigned b = 0; b < 256; ++b) {
        vocab.push_back(sym(static_cast<unsigned char>(b)));
    }
    for (const char* text : {"he", "ll", "hell", "\xC4\xA0w"}) {
        vocab.emplace_back(text);
    }
    const BpeTokenizer parts =
        BpeTokenizer::from_parts(vocab,
                                 {{byte_id('h'), byte_id('e')},
                                  {byte_id('l'), byte_id('l')},
                                  {257, 258},
                                  {byte_id(' '), byte_id('w')}},
                                 {0});
    TITAN_EXPECT_EQ(titan::io::to_json(parts.to_json(), 0), hello_text());

    // A merge whose result is not in the vocabulary.
    TITAN_EXPECT_THROWS(
        BpeTokenizer::from_parts(vocab, {{byte_id('x'), byte_id('y')}}, {0}),
        titan::Error);
}

// --- from_json validation: one test per rule ---

TITAN_TEST(Bpe, FromJsonRejectsNonObject) {
    const std::string msg = from_json_error(std::string("[]"));
    TITAN_EXPECT(contains(msg, "tokenizer.json"));
    TITAN_EXPECT(contains(msg, "object"));
}

TITAN_TEST(Bpe, FromJsonRejectsModelType) {
    const std::string msg = from_json_error(
        replaced(hello_text(), R"("type":"BPE")", R"("type":"WordPiece")"));
    TITAN_EXPECT(contains(msg, "model.type"));
    TITAN_EXPECT(contains(msg, "WordPiece"));
    TITAN_EXPECT(contains(msg, "unsupported"));
}

TITAN_TEST(Bpe, FromJsonRejectsUnknownKey) {
    const std::string msg = from_json_error(
        replaced(hello_text(), R"("fuse_unk")", R"("fuse":1,"fuse_unk")"));
    TITAN_EXPECT(contains(msg, "model.fuse"));
    TITAN_EXPECT(contains(msg, "unknown key"));
}

TITAN_TEST(Bpe, FromJsonRejectsVersion) {
    const std::string msg = from_json_error(
        replaced(hello_text(), R"("version":"1.0")", R"("version":"2.0")"));
    TITAN_EXPECT(contains(msg, "version"));
    TITAN_EXPECT(contains(msg, "2.0"));
}

TITAN_TEST(Bpe, FromJsonRejectsNonNullFeatures) {
    const std::string text = hello_text();
    for (const char* path :
         {"normalizer", "post_processor", "truncation", "padding",
          "model.dropout", "model.unk_token", "model.continuing_subword_prefix",
          "model.end_of_word_suffix"}) {
        const std::string_view field =
            std::string_view(path).substr(std::string_view(path).find('.') + 1);
        const std::string key = "\"" + std::string(field) + "\":null";
        const std::string msg = from_json_error(replaced(
            text, key, "\"" + std::string(field) + "\":{\"type\":\"X\"}"));
        TITAN_EXPECT(contains(msg, path));
        TITAN_EXPECT(contains(msg, "unsupported"));
    }
    // Absent is the same as null.
    const std::string no_normalizer =
        replaced(text, R"("normalizer":null,)", "");
    TITAN_EXPECT_EQ(from_json_error(no_normalizer), std::string());
}

TITAN_TEST(Bpe, FromJsonRejectsTrueFlags) {
    const std::string text = hello_text();
    for (const char* field : {"fuse_unk", "byte_fallback", "ignore_merges"}) {
        const std::string key = "\"" + std::string(field) + "\":";
        const std::string msg =
            from_json_error(replaced(text, key + "false", key + "true"));
        TITAN_EXPECT(contains(msg, "model." + std::string(field)));
        TITAN_EXPECT(contains(msg, "unsupported"));
    }
}

TITAN_TEST(Bpe, FromJsonRejectsPreTokenizer) {
    const std::string text = hello_text();
    std::string msg = from_json_error(replaced(
        text, R"("individual_digits":true)", R"("individual_digits":false)"));
    TITAN_EXPECT(
        contains(msg, "pre_tokenizer.pretokenizers[0].individual_digits"));
    TITAN_EXPECT(contains(msg, "unsupported"));

    msg = from_json_error(
        replaced(text, R"("pre_tokenizer":{)", R"("pre_tokenizer":{"x":1,)"));
    TITAN_EXPECT(contains(msg, "pre_tokenizer.x"));

    msg = from_json_error(replaced(text, R"("type":"Sequence",)", ""));
    TITAN_EXPECT(contains(msg, "pre_tokenizer.type"));
    TITAN_EXPECT(contains(msg, "missing"));
}

TITAN_TEST(Bpe, FromJsonRejectsDecoder) {
    const std::string msg = from_json_error(
        replaced(hello_text(),
                 R"("decoder":{"type":"ByteLevel","add_prefix_space":true)",
                 R"("decoder":{"type":"ByteLevel","add_prefix_space":false)"));
    TITAN_EXPECT(contains(msg, "decoder.add_prefix_space"));
    TITAN_EXPECT(contains(msg, "unsupported"));

    JsonValue doc = hello_doc();
    doc.set("decoder", JsonValue());
    TITAN_EXPECT(contains(from_json_error(doc), "decoder"));
}

TITAN_TEST(Bpe, FromJsonRejectsVocabShape) {
    const std::string text = hello_text();
    std::string msg =
        from_json_error(replaced(text, R"("he":257)", R"("he":"x")"));
    TITAN_EXPECT(contains(msg, "model.vocab"));
    TITAN_EXPECT(contains(msg, "integer"));

    msg = from_json_error(replaced(text, R"("he":257)", R"("he":257.5)"));
    TITAN_EXPECT(contains(msg, "model.vocab"));

    JsonValue doc = hello_doc();
    JsonValue model = doc.at("model");
    model.set("vocab", JsonValue::array());
    doc.set("model", std::move(model));
    msg = from_json_error(doc);
    TITAN_EXPECT(contains(msg, "model.vocab"));
    TITAN_EXPECT(contains(msg, "object"));
}

TITAN_TEST(Bpe, FromJsonRejectsDuplicateId) {
    const std::string msg =
        from_json_error(replaced(hello_text(), R"("he":257)", R"("he":7)"));
    TITAN_EXPECT(contains(msg, "model.vocab"));
    TITAN_EXPECT(contains(msg, "duplicate id 7"));
}

TITAN_TEST(Bpe, FromJsonRejectsIdOutOfRange) {
    const std::string text = hello_text();
    std::string msg =
        from_json_error(replaced(text, R"("he":257)", R"("he":261)"));
    TITAN_EXPECT(contains(msg, "model.vocab"));
    TITAN_EXPECT(contains(msg, "id 261"));
    TITAN_EXPECT(contains(msg, "[0, 261)"));

    msg = from_json_error(replaced(text, R"("he":257)", R"("he":-1)"));
    TITAN_EXPECT(contains(msg, "id -1"));
}

TITAN_TEST(Bpe, FromJsonRejectsEmptyVocab) {
    const std::string msg =
        from_json_error(R"({"model":{"type":"BPE","vocab":{},"merges":[]},)"
                        R"("pre_tokenizer":)" +
                        std::string(kPreTokenizer) + R"(,"decoder":)" +
                        std::string(kDecoder) + "}");
    TITAN_EXPECT(contains(msg, "model.vocab"));
    TITAN_EXPECT(contains(msg, "empty"));
}

TITAN_TEST(Bpe, FromJsonRejectsNonByteLevelEntry) {
    // A raw space is not a byte-level symbol (the space is "Ġ").
    const std::string msg =
        from_json_error(replaced(hello_text(), R"("ll":258)", R"("l l":258)"));
    TITAN_EXPECT(contains(msg, "model.vocab[\"l l\"]"));
    TITAN_EXPECT(contains(msg, "byte-level"));
}

TITAN_TEST(Bpe, FromJsonRejectsMalformedMerge) {
    const std::string text = hello_text();
    // Strings need exactly one space between two non-empty parts.
    for (const char* bad : {R"("h  e")", R"("he")", R"("h e x")", R"(" h")"}) {
        const std::string msg =
            from_json_error(replaced(text, R"("l l")", bad));
        TITAN_EXPECT(contains(msg, "model.merges[1]"));
        TITAN_EXPECT(contains(msg, "exactly one space"));
    }
    // Anything else must be an array of two strings.
    for (const char* bad :
         {R"(["h"])", R"(["h","e","x"])", R"(["h",1])", "7"}) {
        const std::string msg =
            from_json_error(replaced(text, R"("l l")", bad));
        TITAN_EXPECT(contains(msg, "model.merges[1]"));
        TITAN_EXPECT(contains(msg, R"(expected "a b" or ["a", "b"])"));
    }
}

TITAN_TEST(Bpe, FromJsonRejectsMergeNotInVocab) {
    std::string msg = from_json_error(
        make_doc({"he", "ll", "hell", "\xC4\xA0w"},
                 string_merges({"h e", "l l", "he ll", "\xC4\xA0 x"})));
    TITAN_EXPECT(contains(msg, "model.merges[3]"));
    TITAN_EXPECT(contains(msg, "not in the vocabulary"));

    msg = from_json_error(make_doc({"he", "ll", "hell", "\xC4\xA0w"},
                                   string_merges({"h e", "l q2"})));
    TITAN_EXPECT(contains(msg, "model.merges[1]"));
    TITAN_EXPECT(contains(msg, "not in the vocabulary"));
}

TITAN_TEST(Bpe, FromJsonRejectsMergeOfAddedToken) {
    const std::string msg = from_json_error(
        make_doc({"<|endoftext|>a"}, string_merges({"<|endoftext|> a"})));
    TITAN_EXPECT(contains(msg, "model.merges[0]"));
    TITAN_EXPECT(contains(msg, "added token"));
}

TITAN_TEST(Bpe, FromJsonRejectsDuplicateMerge) {
    const std::string msg =
        from_json_error(make_doc({"he"}, string_merges({"h e", "h e"})));
    TITAN_EXPECT(contains(msg, "model.merges[1]"));
    TITAN_EXPECT(contains(msg, "duplicate"));
}

TITAN_TEST(Bpe, FromJsonRejectsAddedTokenFlags) {
    const std::string text = hello_text();
    const std::pair<const char*, const char*> cases[] = {
        {R"("special":true)", R"("special":false)"},
        {R"("normalized":false)", R"("normalized":true)"},
        {R"("lstrip":false)", R"("lstrip":true)"},
        {R"("rstrip":false)", R"("rstrip":true)"},
        {R"("single_word":false)", R"("single_word":true)"},
    };
    for (const auto& [from, to] : cases) {
        const std::string msg = from_json_error(replaced(text, from, to));
        const std::string_view key(from);
        const std::string field(key.substr(1, key.find(':') - 2));
        TITAN_EXPECT(contains(msg, "added_tokens[0]." + field));
        TITAN_EXPECT(contains(msg, "unsupported"));
    }
}

TITAN_TEST(Bpe, FromJsonRejectsAddedTokenId) {
    const std::string text = hello_text();
    std::string msg =
        from_json_error(replaced(text, R"({"id":0,)", R"({"id":5,)"));
    TITAN_EXPECT(contains(msg, "added_tokens[0].id"));
    TITAN_EXPECT(contains(msg, "5 does not match the vocabulary id 0"));

    msg = from_json_error(replaced(text, R"("content":"<|endoftext|>")",
                                   R"("content":"<|end|>")"));
    TITAN_EXPECT(contains(msg, "added_tokens[0].content"));
    TITAN_EXPECT(contains(msg, "not in the vocabulary"));
}

TITAN_TEST(Bpe, FromJsonRejectsEmptyAddedToken) {
    const std::string msg = from_json_error(replaced(
        hello_text(), R"("content":"<|endoftext|>")", R"("content":"")"));
    TITAN_EXPECT(contains(msg, "added_tokens[0].content"));
    TITAN_EXPECT(contains(msg, "is empty"));
}

TITAN_TEST(Bpe, FromJsonRejectsAddedByteSymbol) {
    // "a" (id 98) is the symbol of byte 0x61; as an added token it would
    // take over that byte inside BPE in the reference.
    const std::string msg = from_json_error(
        replaced(hello_text(), R"({"id":0,"content":"<|endoftext|>")",
                 R"({"id":98,"content":"a")"));
    TITAN_EXPECT(contains(msg, "added_tokens[0].content"));
    TITAN_EXPECT(contains(msg, "symbol of a single byte"));
}

TITAN_TEST(Bpe, FromJsonRejectsMergeBeforeItsPart) {
    // The reviewer's file: "ab a" uses ab before "a b" creates it. The
    // reference encodes "abab" as aba, b; merging by rank over all
    // occurrences would give ab, ab, so the file is refused.
    std::string msg = from_json_error(
        make_doc({"ab", "aba"}, string_merges({"ab a", "a b"})));
    TITAN_EXPECT(contains(msg, "model.merges[0]"));
    TITAN_EXPECT(contains(msg, "part \"ab\""));
    TITAN_EXPECT(contains(msg, "earlier merge"));

    // A part that no merge creates at all.
    msg = from_json_error(make_doc({"ab", "abc"}, string_merges({"ab c"})));
    TITAN_EXPECT(contains(msg, "model.merges[0]"));
    TITAN_EXPECT(contains(msg, "part \"ab\""));

    // The same order through from_parts: ab = 257, aba = 258.
    std::vector<std::string> vocab = {"<|endoftext|>"};
    for (unsigned b = 0; b < 256; ++b) {
        vocab.push_back(sym(static_cast<unsigned char>(b)));
    }
    vocab.emplace_back("ab");
    vocab.emplace_back("aba");
    TITAN_EXPECT_THROWS(
        BpeTokenizer::from_parts(
            vocab, {{257, byte_id('a')}, {byte_id('a'), byte_id('b')}}, {0}),
        titan::Error);
    // In the right order the same parts are accepted.
    const BpeTokenizer ok = BpeTokenizer::from_parts(
        vocab, {{byte_id('a'), byte_id('b')}, {257, byte_id('a')}}, {0});
    TITAN_EXPECT(ok.encode("aba") == ids({258}));
    TITAN_EXPECT(ok.encode("abab") == ids({257, 257}));
}
