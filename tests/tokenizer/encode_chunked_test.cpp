// encode_chunked must give exactly the ids of Tokenizer::encode.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/parallel.hpp"
#include "harness/test.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "support/random_text.hpp"
#include "tokenizer/char_tokenizer.hpp"
#include "tokenizer/encode_chunked.hpp"
#include "tokenizer/tokenizer.hpp"

namespace {

using titan::tokenizer::Tokenizer;

const std::filesystem::path kFixtures(TITAN_FIXTURES_DIR);
const std::string kMarker = "<|endoftext|>";

std::unique_ptr<Tokenizer> load(const std::filesystem::path& path) {
    return titan::tokenizer::load_tokenizer(
        titan::io::parse_json(titan::io::read_file(path)));
}

std::string corpus() {
    return titan::io::read_file(kFixtures / "bpe" / "corpus.txt");
}

// `text` with "\n<|endoftext|>\n" after every 7th line.
std::string with_markers(const std::string& text, bool trailing_marker) {
    std::string out;
    int lines = 0;
    for (const char c : text) {
        out += c;
        if (c == '\n' && ++lines % 7 == 0) out += "\n" + kMarker + "\n";
    }
    if (trailing_marker) out += kMarker;
    return out;
}

// `text` with every "\n" replaced by "\r\n".
std::string crlf(const std::string& text) {
    std::string out;
    for (const char c : text) {
        if (c == '\n') out += '\r';
        out += c;
    }
    return out;
}

// Restores the thread count when a test ends.
struct ThreadCount {
    int saved = titan::num_threads();
    ~ThreadCount() { titan::set_num_threads(saved); }
};

}  // namespace

TITAN_TEST(EncodeChunked, MatchesWholeEncode) {
    std::vector<std::string> texts = {"", kMarker, "a" + kMarker,
                                      kMarker + kMarker + "b"};
    const std::string base = corpus();
    texts.push_back(with_markers(base, false));
    texts.push_back(with_markers(base, true));
    texts.push_back(crlf(with_markers(base, true)));
    std::string joined;
    for (const std::string& s : titan::test::random_texts(7, 50)) {
        if (!joined.empty()) joined += kMarker;
        joined += s;
    }
    texts.push_back(joined);

    for (const char* name : {"bpe/trained.json", "smollm2/tokenizer.json"}) {
        const auto tok = load(kFixtures / name);
        for (const std::string& text : texts) {
            TITAN_EXPECT(titan::tokenizer::encode_chunked(*tok, text) ==
                         tok->encode(text));
        }
    }
}

TITAN_TEST(EncodeChunked, ThreadCountDoesNotMatter) {
    const ThreadCount restore;
    const auto tok = load(kFixtures / "bpe" / "trained.json");
    const std::string text = with_markers(corpus(), true);
    // With 8 threads the text must be split into enough ranges for
    // parallel_for to start workers (at least 2 per thread).
    TITAN_EXPECT(titan::tokenizer::split_at_end_of_text(text, 32).size() >= 16);
    titan::set_num_threads(1);
    const std::vector<std::int32_t> one =
        titan::tokenizer::encode_chunked(*tok, text);
    titan::set_num_threads(8);
    const std::vector<std::int32_t> eight =
        titan::tokenizer::encode_chunked(*tok, text);
    TITAN_EXPECT(!one.empty());
    TITAN_EXPECT(one == eight);
}

TITAN_TEST(EncodeChunked, RangesCoverTheTextAtMarkers) {
    const std::string text = with_markers(corpus(), true);
    const auto ranges = titan::tokenizer::split_at_end_of_text(text, 32);
    TITAN_EXPECT(ranges.size() >= 16);
    std::size_t next = 0;
    for (const auto& [begin, end] : ranges) {
        TITAN_EXPECT_EQ(begin, next);
        TITAN_EXPECT(end > begin);
        TITAN_EXPECT(
            end == text.size() ||
            text.compare(end - kMarker.size(), kMarker.size(), kMarker) == 0);
        next = end;
    }
    TITAN_EXPECT_EQ(next, text.size());
}

TITAN_TEST(EncodeChunked, CharTokenizerEncodesWhole) {
    const auto tok = titan::tokenizer::CharTokenizer::build("ab<|endoftext|>");
    const std::string text = "ab<|endoftext|>ba";
    TITAN_EXPECT(titan::tokenizer::encode_chunked(tok, text) ==
                 tok.encode(text));
}

TITAN_TEST(EncodeChunked, RethrowsEncodeErrors) {
    const auto tok = titan::tokenizer::CharTokenizer::build("ab");
    TITAN_EXPECT_THROWS(
        titan::tokenizer::encode_chunked(tok, "ab<|endoftext|>c"),
        titan::Error);
}

TITAN_TEST(EncodeChunked, RethrowsErrorsFromWorkerThreads) {
    const ThreadCount restore;
    titan::set_num_threads(4);
    const auto tok = load(kFixtures / "bpe" / "trained.json");
    // 4 threads get 16 ranges, 4 each; the last block runs on the calling
    // thread, so a bad byte at the start of range 4 (the second block) makes
    // a worker thread throw.
    std::string text = with_markers(corpus(), true);
    const auto ranges = titan::tokenizer::split_at_end_of_text(text, 16);
    TITAN_EXPECT(ranges.size() >= 8);
    text[ranges[ranges.size() / 4].first] = '\xff';
    TITAN_EXPECT_THROWS(titan::tokenizer::encode_chunked(*tok, text),
                        titan::Error);
}
