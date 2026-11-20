#include "core/utf8.hpp"

#include "core/error.hpp"

namespace titan {

bool decode_utf8_at(std::string_view text, std::size_t pos,
                    char32_t& code_point, std::size_t& length) {
    const auto b0 = static_cast<unsigned char>(text[pos]);
    std::size_t len = 0;
    char32_t cp = 0;
    char32_t min_cp = 0;  // smallest value that needs this many bytes
    if (b0 < 0x80) {
        code_point = b0;
        length = 1;
        return true;
    } else if (b0 >= 0xC0 && b0 < 0xE0) {
        len = 2;
        cp = b0 & 0x1Fu;
        min_cp = 0x80;
    } else if (b0 >= 0xE0 && b0 < 0xF0) {
        len = 3;
        cp = b0 & 0x0Fu;
        min_cp = 0x800;
    } else if (b0 >= 0xF0 && b0 < 0xF8) {
        len = 4;
        cp = b0 & 0x07u;
        min_cp = 0x10000;
    } else {
        return false;  // stray continuation byte or 0xF8..0xFF
    }
    if (text.size() - pos < len) return false;  // truncated
    for (std::size_t i = 1; i < len; ++i) {
        const auto b = static_cast<unsigned char>(text[pos + i]);
        if ((b & 0xC0) != 0x80) return false;
        cp = (cp << 6) | (b & 0x3Fu);
    }
    if (cp < min_cp) return false;                   // overlong
    if (cp >= 0xD800 && cp <= 0xDFFF) return false;  // surrogate
    if (cp > 0x10FFFF) return false;
    code_point = cp;
    length = len;
    return true;
}

std::vector<char32_t> decode_utf8(std::string_view text) {
    std::vector<char32_t> out;
    out.reserve(text.size());
    std::size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = 0;
        std::size_t len = 0;
        TITAN_CHECK(
            decode_utf8_at(text, pos, cp, len),
            "decode_utf8: invalid UTF-8 at byte offset " + std::to_string(pos));
        out.push_back(cp);
        pos += len;
    }
    return out;
}

std::string encode_utf8(char32_t cp) {
    TITAN_CHECK(cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF),
                "encode_utf8: " + std::to_string(static_cast<unsigned>(cp)) +
                    " is not a Unicode scalar value");
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

bool is_valid_utf8(std::string_view text) {
    std::size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = 0;
        std::size_t len = 0;
        if (!decode_utf8_at(text, pos, cp, len)) return false;
        pos += len;
    }
    return true;
}

}  // namespace titan
