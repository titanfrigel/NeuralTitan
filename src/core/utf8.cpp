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

std::size_t utf8_complete_prefix(std::string_view bytes) {
    const std::size_t n = bytes.size();
    // The lead byte of an incomplete sequence is one of the last 3 bytes, and
    // only continuation bytes follow it.
    for (std::size_t back = 1; back <= 3 && back <= n; ++back) {
        const std::size_t start = n - back;
        const auto b0 = static_cast<unsigned char>(bytes[start]);
        if ((b0 & 0xC0) == 0x80) continue;  // continuation: look further back
        // The sequence length the lead byte announces. C0, C1 (always
        // overlong) and F5..FF (above U+10FFFF or never valid) start no valid
        // sequence, the same rules decode_utf8_at applies.
        std::size_t len = 0;
        if (b0 >= 0xC2 && b0 <= 0xDF) {
            len = 2;
        } else if (b0 >= 0xE0 && b0 <= 0xEF) {
            len = 3;
        } else if (b0 >= 0xF0 && b0 <= 0xF4) {
            len = 4;
        } else {
            return n;  // ASCII or an invalid lead
        }
        if (back >= len) return n;  // the sequence is not cut short
        if (back >= 2) {
            // The second byte's range for a valid sequence (Unicode table
            // 3-7): E0 needs A0..BF (else overlong), ED 80..9F (else a
            // surrogate), F0 90..BF (else overlong), F4 80..8F (else above
            // U+10FFFF); the rest 80..BF, which the loop already ensured.
            const auto b1 = static_cast<unsigned char>(bytes[start + 1]);
            const bool valid = (b0 == 0xE0)   ? b1 >= 0xA0
                               : (b0 == 0xED) ? b1 <= 0x9F
                               : (b0 == 0xF0) ? b1 >= 0x90
                               : (b0 == 0xF4) ? b1 <= 0x8F
                                              : true;
            if (!valid) return n;
        }
        return start;
    }
    return n;  // only continuation bytes at the end: invalid, so complete
}

std::string Utf8Stream::push(std::string_view bytes) {
    pending_ += bytes;
    const std::size_t n = utf8_complete_prefix(pending_);
    std::string complete = pending_.substr(0, n);
    pending_.erase(0, n);
    return complete;
}

std::string Utf8Stream::finish() {
    std::string rest;
    rest.swap(pending_);
    return rest;
}

}  // namespace titan
