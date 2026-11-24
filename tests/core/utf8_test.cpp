#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/utf8.hpp"
#include "harness/test.hpp"

TITAN_TEST(Utf8, RoundTripBoundaries) {
    const char32_t cps[] = {0x0,    0x41,    0x7F,     0x80,   0x7FF, 0x800,
                            0xFFFF, 0x10000, 0x10FFFF, 0xD7FF, 0xE000};
    for (const char32_t cp : cps) {
        const std::string s = titan::encode_utf8(cp);
        const std::vector<char32_t> back = titan::decode_utf8(s);
        TITAN_EXPECT_EQ(back.size(), std::size_t{1});
        TITAN_EXPECT(back[0] == cp);
        TITAN_EXPECT(titan::is_valid_utf8(s));
    }
    TITAN_EXPECT_EQ(titan::encode_utf8(0xE9), std::string("\xC3\xA9"));
    TITAN_EXPECT_EQ(titan::encode_utf8(0x1F600),
                    std::string("\xF0\x9F\x98\x80"));
    const std::vector<char32_t> mixed =
        titan::decode_utf8("A\xC3\xA9\xE2\x82\xAC");
    TITAN_EXPECT_EQ(mixed.size(), std::size_t{3});
    TITAN_EXPECT(mixed[2] == 0x20AC);
}

TITAN_TEST(Utf8, RejectsInvalid) {
    const std::string bad[] = {
        "\x80",              // stray continuation
        "\xC3",              // truncated 2-byte
        "\xE2\x82",          // truncated 3-byte
        "\xF0\x9F\x98",      // truncated 4-byte
        "\xC3\x28",          // bad continuation
        "\xC0\xAF",          // overlong '/'
        "\xE0\x80\xAF",      // overlong 3-byte
        "\xF0\x80\x80\xAF",  // overlong 4-byte
        "\xED\xA0\x80",      // surrogate D800
        "\xED\xBF\xBF",      // surrogate DFFF
        "\xF4\x90\x80\x80",  // above U+10FFFF
        "\xF8\x88\x80\x80\x80",
        "\xFF",
    };
    for (const std::string& s : bad) {
        TITAN_EXPECT(!titan::is_valid_utf8(s));
        TITAN_EXPECT_THROWS(titan::decode_utf8(s), titan::Error);
    }
    try {
        titan::decode_utf8("ab\xC3");
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(std::string(e.what()).find("offset 2") !=
                     std::string::npos);
    }
    TITAN_EXPECT_THROWS(titan::encode_utf8(0xD800), titan::Error);
    TITAN_EXPECT_THROWS(titan::encode_utf8(0x110000), titan::Error);
}

// A trailing lead byte with too few continuation bytes is held back; complete
// text and invalid bytes are not.
TITAN_TEST(Utf8, CompletePrefix) {
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("ab"), std::size_t{2});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("a\xC3"), std::size_t{1});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("a\xE6\x97"), std::size_t{1});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("\xF0\x9F\x99"),
                    std::size_t{0});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("\xF0\x9F\x99\x82"),
                    std::size_t{4});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("a\xFF"), std::size_t{2});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("\x80"), std::size_t{1});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix(""), std::size_t{0});
    // Prefixes no full sequence can start with count as complete: an
    // overlong lead, a surrogate, a value above U+10FFFF.
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("a\xC0"), std::size_t{2});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("\xED\xA0"), std::size_t{2});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("\xF4\x90"), std::size_t{2});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("\xE0\x80"), std::size_t{2});
    TITAN_EXPECT_EQ(titan::utf8_complete_prefix("\xF5"), std::size_t{1});
}
