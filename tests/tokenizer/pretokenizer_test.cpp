#include <string>
#include <string_view>
#include <vector>

#include "core/error.hpp"
#include "harness/test.hpp"
#include "support/random_text.hpp"
#include "tokenizer/pretokenizer.hpp"
#include "tokenizer/unicode_tables.hpp"

namespace {

using titan::tokenizer::is_letter;
using titan::tokenizer::is_number;
using titan::tokenizer::is_numeric;
using titan::tokenizer::is_whitespace;
using titan::tokenizer::pretokenize;

std::vector<std::string> pieces(std::string_view text) {
    std::vector<std::string> out;
    for (const std::string_view piece : pretokenize(text)) {
        out.emplace_back(piece);
    }
    return out;
}

}  // namespace

TITAN_TEST(Unicode, Classes) {
    TITAN_EXPECT(is_letter(U'a'));
    TITAN_EXPECT(is_letter(0x00E9));  // é, Ll
    TITAN_EXPECT(is_letter(0x65E5));  // 日, Lo
    TITAN_EXPECT(is_letter(0x0645));  // Arabic meem, Lo

    TITAN_EXPECT(is_number(U'7'));
    TITAN_EXPECT(is_number(0x0661));  // Arabic-Indic one, Nd
    TITAN_EXPECT(is_number(0x216B));  // Roman numeral twelve, Nl
    TITAN_EXPECT(is_number(0x00BD));  // one half, No

    TITAN_EXPECT(is_whitespace(0x00A0));  // no-break space
    TITAN_EXPECT(is_whitespace(0x3000));  // ideographic space
    TITAN_EXPECT(is_whitespace(0x2028));  // line separator

    for (const char32_t cp : {char32_t{U'_'}, char32_t{0x1F642}}) {
        TITAN_EXPECT(!is_letter(cp));
        TITAN_EXPECT(!is_number(cp));
        TITAN_EXPECT(!is_whitespace(cp));
    }
}

TITAN_TEST(Unicode, MatchesReferenceVersion) {
    // Unicode 16 additions that Python's unicodedata (15.1) does not know but
    // the reference classifies: numbers isolated by the Digits stage, and
    // letters matched by \p{L}.
    TITAN_EXPECT(is_numeric(0x1CCF1));  // outlined digit one
    TITAN_EXPECT(is_numeric(0x16D71));  // Kirat Rai digit one
    TITAN_EXPECT(is_numeric(0x10D41));  // Garay digit one
    TITAN_EXPECT(is_numeric(0x1E5F1));  // Ol Onal digit one
    TITAN_EXPECT(is_letter(0x11380));   // Tulu-Tigalari letter a
    TITAN_EXPECT(is_letter(0x10D50));   // Garay capital letter
    // The two number classes differ: the Digits stage (Rust's
    // char::is_numeric) knows the digits U+11DE0..U+11DE9, added after
    // Unicode 16; Oniguruma's \p{N} does not.
    TITAN_EXPECT(is_numeric(0x11DE0));
    TITAN_EXPECT(!is_number(0x11DE0));
}

TITAN_TEST(Pretokenize, Reference) {
    // Observed from Hugging Face tokenizers with SmolLM2's pre-tokenizer.
    const std::vector<std::string> expected = {"x", " ", " '", " ",
                                               "1", "2", "ab"};
    TITAN_EXPECT(pieces("x  ' 12ab") == expected);
}

TITAN_TEST(Pretokenize, Contractions) {
    const std::vector<std::string> lower = {"I", "'m", " don", "'t"};
    TITAN_EXPECT(pieces("I'm don't") == lower);
    // The contractions are case-sensitive: "'M" is punctuation then a letter.
    const std::vector<std::string> upper = {"I", "'", "M"};
    TITAN_EXPECT(pieces("I'M") == upper);
}

TITAN_TEST(Pretokenize, WhitespaceRuns) {
    // A run before a word leaves its last space to the word; a run at the end
    // stays whole.
    const std::vector<std::string> expected = {"a", "\n\n ", " b", "  "};
    TITAN_EXPECT(pieces("a\n\n  b  ") == expected);
}

TITAN_TEST(Pretokenize, DigitsAreIndividual) {
    const std::vector<std::string> ascii = {"ab", "1", "2", "3"};
    TITAN_EXPECT(pieces("ab123") == ascii);
    // x, then U+0661 and U+0662 (Arabic-Indic one and two).
    const std::vector<std::string> arabic = {"x", "\xD9\xA1", "\xD9\xA2"};
    TITAN_EXPECT(pieces("x\xD9\xA1\xD9\xA2") == arabic);
}

TITAN_TEST(Pretokenize, Unicode16) {
    // Observed from Hugging Face tokenizers: Unicode 16 digits are split one
    // per piece, and Unicode 16 letters join the letter run.
    const std::vector<std::string> outlined = {"\xF0\x9C\xB3\xB1",
                                               "\xF0\x9C\xB3\xB2"};
    TITAN_EXPECT(pieces("\xF0\x9C\xB3\xB1\xF0\x9C\xB3\xB2") == outlined);
    const std::vector<std::string> kirat = {"\xF0\x96\xB5\xB1",
                                            "\xF0\x96\xB5\xB2"};
    TITAN_EXPECT(pieces("\xF0\x96\xB5\xB1\xF0\x96\xB5\xB2") == kirat);
    // a, U+11380 or U+10D50, then b written as \x62, since a literal b would
    // extend the hex escape before it.
    const std::string tulu = "a\xF0\x91\x8E\x80\x62";
    TITAN_EXPECT(pieces(tulu) == std::vector<std::string>{tulu});
    const std::string garay = "a\xF0\x90\xB5\x90\x62";
    TITAN_EXPECT(pieces(garay) == std::vector<std::string>{garay});
}

TITAN_TEST(Pretokenize, ConcatenationIsInput) {
    for (const std::string& text : titan::test::random_texts(7, 200)) {
        std::string joined;
        bool any_empty = false;
        for (const std::string_view piece : pretokenize(text)) {
            any_empty = any_empty || piece.empty();
            joined += piece;
        }
        TITAN_EXPECT_EQ(joined, text);
        TITAN_EXPECT(!any_empty);
    }
}

TITAN_TEST(Pretokenize, RejectsInvalidUtf8) {
    std::string msg;
    try {
        pretokenize("ab\xff");
    } catch (const titan::Error& e) {
        msg = e.what();
    }
    TITAN_EXPECT(msg.find("byte offset 2") != std::string::npos);

    // The offset counts bytes, not code points: the two bytes of U+00E9
    // put \xFF at byte 2, though it is the second code point.
    msg.clear();
    try {
        pretokenize("\xC3\xA9\xFF");
    } catch (const titan::Error& e) {
        msg = e.what();
    }
    TITAN_EXPECT(msg.find("byte offset 2") != std::string::npos);
}
