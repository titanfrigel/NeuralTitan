// BPE training (docs/math/bpe.md, "Training"): the incremental trainer
// against the hand derivation and the naive trainer, its stopping rules, and
// the committed trained fixture against Hugging Face tokenizers.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "harness/test.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "support/random_text.hpp"
#include "tokenizer/bpe_tokenizer.hpp"
#include "tokenizer/bpe_trainer.hpp"

namespace {

using titan::io::JsonValue;
using titan::tokenizer::bpe_merges;
using titan::tokenizer::bpe_merges_naive;
using titan::tokenizer::BpeMerge;
using titan::tokenizer::BpeTokenizer;
using titan::tokenizer::train_bpe;

const std::filesystem::path kFixtures =
    std::filesystem::path(TITAN_FIXTURES_DIR) / "bpe";

// Byte b is id 1 + b in a trained vocabulary.
std::int32_t byte_id(char byte) {
    return 1 + static_cast<std::int32_t>(static_cast<unsigned char>(byte));
}

// "(left right count)" for each merge, so a failure shows the whole list.
std::string show(const std::vector<BpeMerge>& merges) {
    std::string out;
    for (const BpeMerge& m : merges) {
        out += "(" + std::to_string(m.left) + " " + std::to_string(m.right) +
               " " + std::to_string(m.count) + ")";
    }
    return out;
}

const std::string& fixture_corpus() {
    static const std::string corpus =
        titan::io::read_file(kFixtures / "corpus.txt");
    return corpus;
}

// The token each merge creates, as bytes: merge i creates id 257 + i.
std::vector<std::string> merged_texts(const BpeTokenizer& tok) {
    std::vector<std::string> texts;
    for (std::int64_t i = 0; i < tok.num_merges(); ++i) {
        const std::int32_t id = static_cast<std::int32_t>(257 + i);
        texts.push_back(tok.decode(std::vector<std::int32_t>{id}));
    }
    return texts;
}

}  // namespace

TITAN_TEST(BpeTrain, HandExample) {
    // Words: "aaab", "Ġaaab", "Ġab" (Ġ is the leading space), once each.
    // Step 1: (a, a) occurs twice in each of the first two words: 4.
    // Step 2: the words are "aa a b", "Ġ aa a b", "Ġ a b"; (aa, a) has 2 and
    // (a, b) has 3.
    const std::vector<BpeMerge> merges = bpe_merges("aaab aaab ab", 2);
    TITAN_EXPECT_EQ(show(merges), show({{byte_id('a'), byte_id('a'), 4},
                                        {byte_id('a'), byte_id('b'), 3}}));
    TITAN_EXPECT_EQ(show(bpe_merges_naive("aaab aaab ab", 2)), show(merges));
}

TITAN_TEST(BpeTrain, MatchesNaive) {
    // 20 corpora of 200 to 2000 bytes: whole random texts are appended until
    // the corpus reaches a drawn length in [200, 1744]; a text has at most
    // 64 symbols of at most 4 bytes, so the corpus stays below 2000 bytes.
    titan::Rng rng(11);
    for (int c = 0; c < 20; ++c) {
        const std::int64_t target = 200 + rng.randint(1545);
        std::string corpus;
        std::uint64_t seed = rng.next_u64();
        while (static_cast<std::int64_t>(corpus.size()) < target) {
            for (const std::string& text :
                 titan::test::random_texts(seed++, 16)) {
                corpus += text;
                if (static_cast<std::int64_t>(corpus.size()) >= target) {
                    break;
                }
            }
        }
        TITAN_EXPECT(corpus.size() >= 200 && corpus.size() < 2000);
        for (const std::int64_t max_merges : {1, 10, 100}) {
            TITAN_EXPECT_EQ(show(bpe_merges(corpus, max_merges)),
                            show(bpe_merges_naive(corpus, max_merges)));
        }
    }
    // Real text, where many counts tie and change at every step.
    const std::string_view text =
        std::string_view(fixture_corpus()).substr(0, 5000);
    TITAN_EXPECT_EQ(show(bpe_merges(text, 300)),
                    show(bpe_merges_naive(text, 300)));
}

TITAN_TEST(BpeTrain, StopsWhenNoPairRepeats) {
    const BpeTokenizer single = train_bpe("abc", 1000);
    TITAN_EXPECT_EQ(single.num_merges(), 0);
    TITAN_EXPECT_EQ(single.vocab_size(), 257);

    // Words "abab" and "Ġabab": (a, b) has 4; then (ab, ab) has 2; then
    // every pair occurs once.
    const std::vector<BpeMerge> merges = bpe_merges("abab abab", 1000);
    TITAN_EXPECT_EQ(show(merges),
                    show({{byte_id('a'), byte_id('b'), 4}, {257, 257, 2}}));
    TITAN_EXPECT_EQ(train_bpe("abab abab", 1000).vocab_size(), 259);
}

TITAN_TEST(BpeTrain, Deterministic) {
    TITAN_EXPECT_EQ(show(bpe_merges(fixture_corpus(), 300)),
                    show(bpe_merges(fixture_corpus(), 300)));
}

TITAN_TEST(BpeTrain, SkipsEndOfText) {
    // Only "ab" repeats; the bytes of <|endoftext|> are never words.
    TITAN_EXPECT_EQ(show(bpe_merges("ab<|endoftext|>ab", 100)),
                    show({{byte_id('a'), byte_id('b'), 2}}));
    // Kept as text, "endoftext" would repeat and be merged.
    TITAN_EXPECT_EQ(show(bpe_merges("x<|endoftext|><|endoftext|>x", 100)),
                    show({}));

    const BpeTokenizer tok = train_bpe("ab<|endoftext|>ab", 300);
    TITAN_EXPECT_EQ(tok.encode("<|endoftext|>"), std::vector<std::int32_t>{0});
    TITAN_EXPECT(tok.eos_id() == 0);
}

TITAN_TEST(BpeTrain, MergesCreateDistinctTokens) {
    // Ruling R6: no merge recreates an existing token. In the first corpus,
    // "abc" could be formed as ab+c or as a+bc. The tokenizer must also load
    // from its own file, whose loader rejects a repeated result.
    for (const std::string_view corpus :
         {std::string_view("abc abc bcx bcx abx abx abcbc"),
          std::string_view(fixture_corpus())}) {
        const BpeTokenizer tok = train_bpe(corpus, 1000);
        const std::vector<std::string> texts = merged_texts(tok);
        for (std::size_t i = 0; i < texts.size(); ++i) {
            TITAN_EXPECT(texts[i].size() > 1);
            for (std::size_t j = 0; j < i; ++j) {
                if (texts[i] == texts[j]) {
                    titan::test::report_failure(
                        __FILE__, __LINE__,
                        "merges " + std::to_string(j) + " and " +
                            std::to_string(i) + " both create \"" + texts[i] +
                            "\"");
                }
            }
        }
        const BpeTokenizer loaded = BpeTokenizer::from_json(tok.to_json());
        TITAN_EXPECT_EQ(loaded.num_merges(), tok.num_merges());
    }
}

TITAN_TEST(BpeTrain, EncodesCorpusRoundTrip) {
    const BpeTokenizer tok = train_bpe(fixture_corpus(), 512);
    TITAN_EXPECT_EQ(tok.vocab_size(), 512);
    const std::vector<std::int32_t> ids = tok.encode(fixture_corpus());
    TITAN_EXPECT(ids.size() < fixture_corpus().size() / 2);
    TITAN_EXPECT_EQ(tok.decode(ids), fixture_corpus());
}

TITAN_TEST(BpeTrain, RejectsSmallVocab) {
    std::string msg;
    try {
        train_bpe("abc", 256);
    } catch (const titan::Error& e) {
        msg = e.what();
    }
    TITAN_EXPECT(msg.find("256") != std::string::npos);
    TITAN_EXPECT_THROWS(bpe_merges("abc", -1), titan::Error);
}

TITAN_TEST(BpeTrained, FixtureIsTrainerOutput) {
    // tests/fixtures/bpe/trained.json is `titan tokenizer train --type bpe
    // --vocab 512` on corpus.txt; it must stay what the trainer produces.
    const JsonValue fixture =
        titan::io::parse_json(titan::io::read_file(kFixtures / "trained.json"));
    TITAN_EXPECT_EQ(
        titan::io::to_json(fixture, 0),
        titan::io::to_json(train_bpe(fixture_corpus(), 512).to_json(), 0));
}

TITAN_TEST(BpeTrained, MatchesHuggingFace) {
    const BpeTokenizer tok = BpeTokenizer::from_json(titan::io::parse_json(
        titan::io::read_file(kFixtures / "trained.json")));
    const JsonValue doc =
        titan::io::parse_json(titan::io::read_file(kFixtures / "encode.json"));
    const auto& cases = doc.at("cases").as_array();
    TITAN_EXPECT_EQ(cases.size(), std::size_t{26});
    for (std::size_t i = 0; i < cases.size(); ++i) {
        std::vector<std::int32_t> want;
        for (const JsonValue& id : cases[i].at("ids").as_array()) {
            want.push_back(static_cast<std::int32_t>(id.as_int()));
        }
        const std::string& text = cases[i].at("text").as_string();
        if (tok.encode(text) != want) {
            titan::test::report_failure(
                __FILE__, __LINE__,
                "case " + std::to_string(i) + ": ids differ from tokenizers");
        }
    }
}
