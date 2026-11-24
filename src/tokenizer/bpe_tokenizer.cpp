#include "tokenizer/bpe_tokenizer.hpp"

#include <cmath>
#include <limits>
#include <utility>

#include "core/error.hpp"
#include "core/utf8.hpp"
#include "io/json_fields.hpp"
#include "tokenizer/byte_level.hpp"
#include "tokenizer/pretokenizer.hpp"

namespace titan::tokenizer {

namespace {

using io::JsonValue;

constexpr std::string_view kEndOfText = "<|endoftext|>";

// SmolLM2's pre-tokenizer and decoder, the only ones supported
// (docs/math/bpe.md, "File format").
constexpr std::string_view kPreTokenizerJson =
    R"({"type":"Sequence","pretokenizers":[)"
    R"({"type":"Digits","individual_digits":true},)"
    R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,)"
    R"("use_regex":true}]})";
constexpr std::string_view kDecoderJson =
    R"({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,)"
    R"("use_regex":true})";

// Ids and ranks are int32, so counts must stay below 2^31.
constexpr std::size_t kMaxCount =
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());

// --- from_json helpers. Every error reads "tokenizer.json: <path>: ...". ---

[[noreturn]] void fail(const std::string& path, const std::string& message) {
    throw_error(__FILE__, __LINE__, "tokenizer.json: " + path + ": " + message);
}

std::string child(const std::string& path, std::string_view key) {
    return path.empty() ? std::string(key) : path + "." + std::string(key);
}

std::string element(const std::string& path, std::size_t index) {
    return path + "[" + std::to_string(index) + "]";
}

std::string quoted(const std::string& text) { return "\"" + text + "\""; }

// A value as compact JSON, cut short so that a huge object cannot flood the
// message.
std::string show(const JsonValue& v) {
    constexpr std::size_t kMaxShown = 80;
    std::string s = io::to_json(v, 0);
    if (s.size() > kMaxShown) {
        s.resize(kMaxShown);
        s += "...";
    }
    return s;
}

void reject_unknown_keys(const JsonValue& obj, const std::string& path,
                         std::initializer_list<std::string_view> allowed) {
    for (const auto& [key, value] : obj.as_object()) {
        bool known = false;
        for (const std::string_view a : allowed) {
            known = known || key == a;
        }
        if (!known) {
            fail(child(path, key), "unknown key");
        }
    }
}

const JsonValue& require(const JsonValue& obj, const std::string& path,
                         std::string_view key) {
    const JsonValue* v = obj.find(key);
    if (v == nullptr) {
        fail(child(path, key), "missing required field");
    }
    return *v;
}

const JsonValue& require_object(const JsonValue& obj, const std::string& path,
                                std::string_view key) {
    const JsonValue& v = require(obj, path, key);
    if (!v.is_object()) {
        fail(child(path, key),
             std::string("expected an object, got ") + io::json_kind_name(v));
    }
    return v;
}

// A feature we do not implement: it must be null or absent.
void expect_null(const JsonValue& obj, const std::string& path,
                 std::string_view key) {
    const JsonValue* v = obj.find(key);
    if (v != nullptr && !v->is_null()) {
        fail(child(path, key),
             "unsupported value " + show(*v) + ", expected null");
    }
}

// A flag we do not implement: it must be false or absent (its default).
void expect_false(const JsonValue& obj, const std::string& path,
                  std::string_view key) {
    const JsonValue* v = obj.find(key);
    if (v != nullptr && !(v->is_bool() && !v->as_bool())) {
        fail(child(path, key),
             "unsupported value " + show(*v) + ", expected false");
    }
}

// Requires `actual` to equal `expected` field by field, naming the first
// field that differs.
void expect_equal(const JsonValue& actual, const JsonValue& expected,
                  const std::string& path) {
    if (expected.is_object() && actual.is_object()) {
        for (const auto& [key, value] : expected.as_object()) {
            const JsonValue* a = actual.find(key);
            if (a == nullptr) {
                fail(child(path, key),
                     "missing required field, expected " + show(value));
            }
            expect_equal(*a, value, child(path, key));
        }
        for (const auto& [key, value] : actual.as_object()) {
            if (expected.find(key) == nullptr) {
                fail(child(path, key), "unknown key");
            }
        }
        return;
    }
    if (expected.is_array() && actual.is_array() &&
        actual.as_array().size() == expected.as_array().size()) {
        for (std::size_t i = 0; i < expected.as_array().size(); ++i) {
            expect_equal(actual.as_array()[i], expected.as_array()[i],
                         element(path, i));
        }
        return;
    }
    // Scalars, or a kind or length mismatch: compare the written forms.
    if (io::to_json(actual, 0) != io::to_json(expected, 0)) {
        fail(path, "unsupported value " + show(actual) + ", expected " +
                       show(expected));
    }
}

// An integral JSON number in [0, n).
std::int32_t id_in_range(const JsonValue& v, const std::string& path,
                         std::size_t n) {
    if (!v.is_number() || v.as_number() != std::trunc(v.as_number())) {
        fail(path, "expected an integer id, got " + show(v));
    }
    const double d = v.as_number();
    if (d < 0.0 || d >= static_cast<double>(n)) {
        fail(path,
             "id " + show(v) + " is outside [0, " + std::to_string(n) + ")");
    }
    return static_cast<std::int32_t>(d);
}

// The two parts of a merge, written "a b" (exactly one space) or ["a", "b"].
std::pair<std::string, std::string> merge_parts(const JsonValue& v,
                                                const std::string& path) {
    if (v.is_string()) {
        const std::string& s = v.as_string();
        const std::size_t space = s.find(' ');
        if (space == std::string::npos || space == 0 || space + 1 == s.size() ||
            s.find(' ', space + 1) != std::string::npos) {
            fail(path,
                 "expected \"a b\" with exactly one space, got " + show(v));
        }
        return {s.substr(0, space), s.substr(space + 1)};
    }
    if (v.is_array() && v.as_array().size() == 2 &&
        v.as_array()[0].is_string() && v.as_array()[1].is_string()) {
        return {v.as_array()[0].as_string(), v.as_array()[1].as_string()};
    }
    fail(path, "expected \"a b\" or [\"a\", \"b\"], got " + show(v));
}

}  // namespace

std::uint64_t BpeTokenizer::pair_key(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left))
            << 32) |
           static_cast<std::uint32_t>(right);
}

BpeTokenizer BpeTokenizer::from_json(const JsonValue& json) {
    if (!json.is_object()) {
        throw_error(__FILE__, __LINE__,
                    std::string("tokenizer.json: expected an object, got ") +
                        io::json_kind_name(json));
    }
    reject_unknown_keys(
        json, "",
        {"version", "truncation", "padding", "added_tokens", "normalizer",
         "pre_tokenizer", "post_processor", "decoder", "model"});
    if (const JsonValue* version = json.find("version");
        version != nullptr &&
        !(version->is_string() && version->as_string() == "1.0")) {
        fail("version",
             "unsupported value " + show(*version) + ", expected \"1.0\"");
    }
    for (const std::string_view key :
         {"normalizer", "post_processor", "truncation", "padding"}) {
        expect_null(json, "", key);
    }
    expect_equal(require(json, "", "pre_tokenizer"),
                 io::parse_json(kPreTokenizerJson), "pre_tokenizer");
    expect_equal(require(json, "", "decoder"), io::parse_json(kDecoderJson),
                 "decoder");

    // --- model ---
    const std::string model_path = "model";
    const JsonValue& model = require_object(json, "", "model");
    reject_unknown_keys(
        model, model_path,
        {"type", "dropout", "unk_token", "continuing_subword_prefix",
         "end_of_word_suffix", "fuse_unk", "byte_fallback", "ignore_merges",
         "vocab", "merges"});
    const JsonValue& type = require(model, model_path, "type");
    if (!(type.is_string() && type.as_string() == "BPE")) {
        fail("model.type",
             "unsupported value " + show(type) + ", expected \"BPE\"");
    }
    for (const std::string_view key :
         {"dropout", "unk_token", "continuing_subword_prefix",
          "end_of_word_suffix"}) {
        expect_null(model, model_path, key);
    }
    for (const std::string_view key :
         {"fuse_unk", "byte_fallback", "ignore_merges"}) {
        expect_false(model, model_path, key);
    }

    // model.vocab: text -> id, with the ids a permutation of [0, n). With n
    // entries, ids that are unique and inside [0, n) are also dense.
    const std::string vocab_path = "model.vocab";
    const JsonValue::Object& entries =
        require_object(model, model_path, "vocab").as_object();
    if (entries.empty()) {
        fail(vocab_path, "is empty");
    }
    if (entries.size() > kMaxCount) {
        fail(vocab_path, "has " + std::to_string(entries.size()) +
                             " entries, more than 2^31 - 1");
    }
    const std::size_t n = entries.size();
    std::vector<std::string> vocab(n);
    std::vector<bool> seen(n, false);
    std::unordered_map<std::string, std::int32_t> id_of;
    id_of.reserve(n);
    for (const auto& [text, value] : entries) {
        const std::int32_t id =
            id_in_range(value, vocab_path + "[" + quoted(text) + "]", n);
        const auto index = static_cast<std::size_t>(id);
        if (seen[index]) {
            fail(vocab_path, "duplicate id " + std::to_string(id) + " (" +
                                 quoted(vocab[index]) + " and " + quoted(text) +
                                 ")");
        }
        seen[index] = true;
        vocab[index] = text;
        id_of.emplace(text, id);
    }

    // added_tokens: special tokens matched literally before pre-tokenizing.
    std::vector<std::int32_t> added_ids;
    std::vector<bool> is_added(n, false);
    if (const JsonValue* added = json.find("added_tokens");
        added != nullptr && !added->is_null()) {
        if (!added->is_array()) {
            fail("added_tokens", std::string("expected an array, got ") +
                                     io::json_kind_name(*added));
        }
        for (std::size_t i = 0; i < added->as_array().size(); ++i) {
            const JsonValue& token = added->as_array()[i];
            const std::string path = element("added_tokens", i);
            if (!token.is_object()) {
                fail(path, std::string("expected an object, got ") +
                               io::json_kind_name(token));
            }
            reject_unknown_keys(token, path,
                                {"id", "content", "single_word", "lstrip",
                                 "rstrip", "normalized", "special"});
            const JsonValue& content = require(token, path, "content");
            if (!content.is_string()) {
                fail(child(path, "content"),
                     "expected a string, got " + show(content));
            }
            if (content.as_string().empty()) {
                fail(child(path, "content"), "is empty");
            }
            // The reference would use such a token's id inside BPE for the
            // byte; we keep added tokens out of BPE, so the case is refused.
            if (const std::optional<std::string> bytes =
                    symbols_to_bytes(content.as_string());
                bytes && bytes->size() == 1) {
                fail(child(path, "content"),
                     quoted(content.as_string()) +
                         " is the byte-level symbol of a single byte");
            }
            const auto found = id_of.find(content.as_string());
            if (found == id_of.end()) {
                fail(child(path, "content"),
                     quoted(content.as_string()) + " is not in the vocabulary");
            }
            const JsonValue& id = require(token, path, "id");
            if (!id.is_number() ||
                id.as_number() != static_cast<double>(found->second)) {
                fail(child(path, "id"),
                     show(id) + " does not match the vocabulary id " +
                         std::to_string(found->second) + " of " +
                         quoted(content.as_string()));
            }
            const JsonValue& special = require(token, path, "special");
            if (!(special.is_bool() && special.as_bool())) {
                fail(child(path, "special"),
                     "unsupported value " + show(special) + ", expected true");
            }
            for (const std::string_view key :
                 {"normalized", "lstrip", "rstrip", "single_word"}) {
                expect_false(token, path, key);
            }
            const auto index = static_cast<std::size_t>(found->second);
            if (is_added[index]) {
                fail(child(path, "content"),
                     "duplicate added token " + quoted(content.as_string()));
            }
            is_added[index] = true;
            added_ids.push_back(found->second);
        }
    }

    // Every other entry is written in the byte-level alphabet. Not every
    // byte needs a symbol of its own: SmolLM2 has none for 21 bytes, and
    // encode drops them (docs/math/bpe.md, "Encode").
    std::vector<bool> is_byte(n, false);  // a single byte-level symbol
    for (std::size_t id = 0; id < n; ++id) {
        if (is_added[id]) {
            continue;
        }
        const std::optional<std::string> bytes = symbols_to_bytes(vocab[id]);
        if (!bytes || bytes->empty()) {
            fail(vocab_path + "[" + quoted(vocab[id]) + "]",
                 "is not text in the byte-level alphabet");
        }
        is_byte[id] = bytes->size() == 1;
    }

    // model.merges, by rank.
    const std::string merges_path = "model.merges";
    const JsonValue& merges_json = require(model, model_path, "merges");
    if (!merges_json.is_array()) {
        fail(merges_path, std::string("expected an array, got ") +
                              io::json_kind_name(merges_json));
    }
    const JsonValue::Array& merge_entries = merges_json.as_array();
    if (merge_entries.size() > kMaxCount) {
        fail(merges_path, "has more than 2^31 - 1 entries");
    }
    std::vector<std::pair<std::int32_t, std::int32_t>> merges;
    merges.reserve(merge_entries.size());
    std::unordered_map<std::uint64_t, std::size_t> rank_of;
    rank_of.reserve(merge_entries.size());
    // creator[id] is the index of the merge that produces token id, or -1.
    // A multi-symbol part must already have a creator, and no token may have
    // two, so that encode's merge order equals the reference's
    // (docs/math/bpe.md, "Merge order").
    std::vector<std::int64_t> creator(n, -1);
    for (std::size_t i = 0; i < merge_entries.size(); ++i) {
        const std::string path = element(merges_path, i);
        const auto [left, right] = merge_parts(merge_entries[i], path);
        std::int32_t part_ids[3] = {};
        const std::string texts[3] = {left, right, left + right};
        for (int k = 0; k < 3; ++k) {
            const auto found = id_of.find(texts[k]);
            if (found == id_of.end()) {
                fail(path, quoted(texts[k]) + " is not in the vocabulary");
            }
            if (is_added[static_cast<std::size_t>(found->second)]) {
                fail(path, quoted(texts[k]) + " is an added token");
            }
            part_ids[k] = found->second;
        }
        for (int k = 0; k < 2; ++k) {
            const auto part = static_cast<std::size_t>(part_ids[k]);
            if (!is_byte[part] && creator[part] < 0) {
                fail(path, "part " + quoted(texts[k]) +
                               " is not the result of an earlier merge");
            }
        }
        const auto [it, inserted] =
            rank_of.emplace(pair_key(part_ids[0], part_ids[1]), i);
        if (!inserted) {
            fail(path, "duplicate of " + element(merges_path, it->second));
        }
        std::int64_t& result_creator =
            creator[static_cast<std::size_t>(part_ids[2])];
        if (result_creator >= 0) {
            fail(path, "result " + quoted(texts[2]) +
                           " is already created by " +
                           element(merges_path,
                                   static_cast<std::size_t>(result_creator)));
        }
        result_creator = static_cast<std::int64_t>(i);
        merges.emplace_back(part_ids[0], part_ids[1]);
    }

    return from_parts(std::move(vocab), std::move(merges),
                      std::move(added_ids));
}

BpeTokenizer BpeTokenizer::from_parts(
    std::vector<std::string> vocab,
    std::vector<std::pair<std::int32_t, std::int32_t>> merges,
    std::vector<std::int32_t> added_ids) {
    const std::size_t n = vocab.size();
    TITAN_CHECK(n > 0 && n <= kMaxCount,
                "BpeTokenizer::from_parts: the vocabulary size " +
                    std::to_string(n) + " is outside [1, 2^31)");
    TITAN_CHECK(merges.size() <= kMaxCount,
                "BpeTokenizer::from_parts: too many merges");
    const auto in_range = [n](std::int32_t id) {
        return id >= 0 && static_cast<std::size_t>(id) < n;
    };

    BpeTokenizer tok;
    tok.is_added_.assign(n, false);
    for (const std::int32_t id : added_ids) {
        TITAN_CHECK(in_range(id), "BpeTokenizer::from_parts: added id " +
                                      std::to_string(id) +
                                      " is outside the vocabulary");
        const auto index = static_cast<std::size_t>(id);
        TITAN_CHECK(!tok.is_added_[index] && !vocab[index].empty(),
                    "BpeTokenizer::from_parts: added id " + std::to_string(id) +
                        " is repeated or empty");
        tok.is_added_[index] = true;
    }

    tok.bytes_.resize(n);
    tok.ids_.reserve(n);
    for (std::size_t id = 0; id < n; ++id) {
        const auto [it, inserted] =
            tok.ids_.emplace(vocab[id], static_cast<std::int32_t>(id));
        TITAN_CHECK(inserted, "BpeTokenizer::from_parts: token " +
                                  std::to_string(id) + " repeats the text of " +
                                  std::to_string(it->second));
        if (tok.is_added_[id]) {
            const std::optional<std::string> bytes =
                symbols_to_bytes(vocab[id]);
            TITAN_CHECK(!(bytes && bytes->size() == 1),
                        "BpeTokenizer::from_parts: added token " +
                            std::to_string(id) +
                            " is the byte-level symbol of a single byte");
            tok.bytes_[id] = vocab[id];  // an added token decodes literally
            continue;
        }
        std::optional<std::string> bytes = symbols_to_bytes(vocab[id]);
        TITAN_CHECK(bytes && !bytes->empty(),
                    "BpeTokenizer::from_parts: token " + std::to_string(id) +
                        " is not text in the byte-level alphabet");
        tok.bytes_[id] = std::move(*bytes);
    }
    for (unsigned b = 0; b < 256; ++b) {
        const auto it = tok.ids_.find(
            encode_utf8(byte_to_symbol(static_cast<std::uint8_t>(b))));
        const bool has_symbol =
            it != tok.ids_.end() &&
            !tok.is_added_[static_cast<std::size_t>(it->second)];
        tok.byte_ids_[b] = has_symbol ? it->second : kNoSymbol;
    }

    tok.merge_of_.reserve(merges.size());
    // Tokens produced by the merges so far; see from_json.
    std::vector<bool> created(n, false);
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        const auto [left, right] = merges[rank];
        TITAN_CHECK(in_range(left) && in_range(right) &&
                        !tok.is_added_[static_cast<std::size_t>(left)] &&
                        !tok.is_added_[static_cast<std::size_t>(right)],
                    "BpeTokenizer::from_parts: merge " + std::to_string(rank) +
                        " has a part outside the ordinary vocabulary");
        for (const std::int32_t part : {left, right}) {
            const auto index = static_cast<std::size_t>(part);
            TITAN_CHECK(tok.bytes_[index].size() == 1 || created[index],
                        "BpeTokenizer::from_parts: part " +
                            std::to_string(part) + " of merge " +
                            std::to_string(rank) +
                            " is not the result of an earlier merge");
        }
        const auto merged =
            tok.ids_.find(vocab[static_cast<std::size_t>(left)] +
                          vocab[static_cast<std::size_t>(right)]);
        TITAN_CHECK(
            merged != tok.ids_.end() &&
                !tok.is_added_[static_cast<std::size_t>(merged->second)],
            "BpeTokenizer::from_parts: the result of merge " +
                std::to_string(rank) + " is not in the vocabulary");
        TITAN_CHECK(!created[static_cast<std::size_t>(merged->second)],
                    "BpeTokenizer::from_parts: merge " + std::to_string(rank) +
                        " creates token " + std::to_string(merged->second) +
                        ", which an earlier merge already creates");
        created[static_cast<std::size_t>(merged->second)] = true;
        const bool inserted =
            tok.merge_of_
                .emplace(pair_key(left, right),
                         Merge{static_cast<std::int32_t>(rank), merged->second})
                .second;
        TITAN_CHECK(inserted, "BpeTokenizer::from_parts: merge " +
                                  std::to_string(rank) + " is a duplicate");
    }

    tok.vocab_ = std::move(vocab);
    tok.merges_ = std::move(merges);
    tok.added_ = std::move(added_ids);
    return tok;
}

JsonValue BpeTokenizer::to_json() const {
    JsonValue added = JsonValue::array();
    for (const std::int32_t id : added_) {
        JsonValue token = JsonValue::object();
        token.set("id", JsonValue::number(id));
        token.set("content",
                  JsonValue::string(vocab_[static_cast<std::size_t>(id)]));
        token.set("single_word", JsonValue::boolean(false));
        token.set("lstrip", JsonValue::boolean(false));
        token.set("rstrip", JsonValue::boolean(false));
        token.set("normalized", JsonValue::boolean(false));
        token.set("special", JsonValue::boolean(true));
        added.push_back(std::move(token));
    }

    // The texts are distinct, so the members can be handed over at once
    // (set() would search the object for every key).
    JsonValue::Object vocab;
    vocab.reserve(vocab_.size());
    for (std::size_t id = 0; id < vocab_.size(); ++id) {
        vocab.emplace_back(vocab_[id],
                           JsonValue::number(static_cast<double>(id)));
    }
    // Ordinary tokens never hold a raw space (it is the symbol U+0120), so
    // "a b" is unambiguous.
    JsonValue merges = JsonValue::array();
    for (const auto& [left, right] : merges_) {
        merges.push_back(
            JsonValue::string(vocab_[static_cast<std::size_t>(left)] + " " +
                              vocab_[static_cast<std::size_t>(right)]));
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
    model.set("vocab", JsonValue::object(std::move(vocab)));
    model.set("merges", std::move(merges));

    JsonValue doc = JsonValue::object();
    doc.set("version", JsonValue::string("1.0"));
    doc.set("truncation", JsonValue());
    doc.set("padding", JsonValue());
    doc.set("added_tokens", std::move(added));
    doc.set("normalizer", JsonValue());
    doc.set("pre_tokenizer", io::parse_json(kPreTokenizerJson));
    doc.set("post_processor", JsonValue());
    doc.set("decoder", io::parse_json(kDecoderJson));
    doc.set("model", std::move(model));
    return doc;
}

void BpeTokenizer::encode_piece(std::string_view piece,
                                std::vector<std::int32_t>& out) const {
    // One symbol per byte to start with. A byte without a symbol is dropped
    // here, before any merge, as the reference does: its neighbours become
    // adjacent and may merge.
    std::vector<std::int32_t> symbols;
    symbols.reserve(piece.size());
    for (const char c : piece) {
        const std::int32_t id = byte_ids_[static_cast<std::uint8_t>(c)];
        if (id != kNoSymbol) {
            symbols.push_back(id);
        }
    }
    // Repeatedly apply the lowest-ranked merge among the adjacent pairs, to
    // all of its non-overlapping occurrences, left to right.
    while (symbols.size() > 1) {
        const Merge* best = nullptr;
        std::size_t best_at = 0;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it =
                merge_of_.find(pair_key(symbols[i], symbols[i + 1]));
            if (it != merge_of_.end() &&
                (best == nullptr || it->second.rank < best->rank)) {
                best = &it->second;
                best_at = i;
            }
        }
        if (best == nullptr) {
            break;
        }
        const std::int32_t left = symbols[best_at];
        const std::int32_t right = symbols[best_at + 1];
        std::size_t write = 0;
        for (std::size_t read = 0; read < symbols.size();) {
            if (read + 1 < symbols.size() && symbols[read] == left &&
                symbols[read + 1] == right) {
                symbols[write++] = best->merged;
                read += 2;
            } else {
                symbols[write++] = symbols[read++];
            }
        }
        symbols.resize(write);
    }
    out.insert(out.end(), symbols.begin(), symbols.end());
}

std::vector<std::int32_t> BpeTokenizer::encode(std::string_view text) const {
    // Validate once up front, so the error names the offset in `text` rather
    // than in one of its spans.
    for (std::size_t pos = 0; pos < text.size();) {
        char32_t cp = 0;
        std::size_t length = 0;
        TITAN_CHECK(decode_utf8_at(text, pos, cp, length),
                    "BpeTokenizer::encode: invalid UTF-8 at byte offset " +
                        std::to_string(pos));
        pos += length;
    }

    std::vector<std::int32_t> out;
    const auto encode_span = [&](std::string_view span) {
        for (const std::string_view piece : pretokenize(span)) {
            encode_piece(piece, out);
        }
    };
    // Split off added tokens: at each position take the longest added token
    // that matches; the text between matches is ordinary text.
    std::size_t span_start = 0;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::string_view rest = text.substr(pos);
        std::int32_t match = -1;
        std::size_t match_length = 0;
        for (const std::int32_t id : added_) {
            const std::string& content = vocab_[static_cast<std::size_t>(id)];
            if (content.size() > match_length && rest.starts_with(content)) {
                match = id;
                match_length = content.size();
            }
        }
        if (match < 0) {
            ++pos;
            continue;
        }
        encode_span(text.substr(span_start, pos - span_start));
        out.push_back(match);
        pos += match_length;
        span_start = pos;
    }
    encode_span(text.substr(span_start));
    return out;
}

std::string BpeTokenizer::decode(std::span<const std::int32_t> ids) const {
    for (std::size_t i = 0; i < ids.size(); ++i) {
        TITAN_CHECK(
            ids[i] >= 0 && static_cast<std::size_t>(ids[i]) < vocab_.size(),
            "BpeTokenizer::decode: id " + std::to_string(ids[i]) +
                " at position " + std::to_string(i) +
                " is outside the vocabulary of size " +
                std::to_string(vocab_.size()));
    }
    std::string out;
    for (const std::int32_t id : ids) {
        out += bytes_[static_cast<std::size_t>(id)];
    }
    return out;
}

std::int64_t BpeTokenizer::vocab_size() const {
    return static_cast<std::int64_t>(vocab_.size());
}

std::optional<std::int32_t> BpeTokenizer::eos_id() const {
    for (const std::int32_t id : added_) {
        if (vocab_[static_cast<std::size_t>(id)] == kEndOfText) {
            return id;
        }
    }
    return std::nullopt;
}

std::int64_t BpeTokenizer::num_merges() const {
    return static_cast<std::int64_t>(merges_.size());
}

}  // namespace titan::tokenizer
