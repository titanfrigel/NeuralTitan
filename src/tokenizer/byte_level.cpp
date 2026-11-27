#include "tokenizer/byte_level.hpp"

#include <array>

#include "core/utf8.hpp"

namespace titan::tokenizer {

namespace {

// Bytes that GPT-2 considers printable keep their own code point.
constexpr bool keeps_code_point(unsigned byte) {
    return (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
           (byte >= 174 && byte <= 255);
}

// The 68 other bytes are numbered in increasing order and shifted past the
// Latin-1 block: the n-th of them maps to U+0100 + n.
constexpr std::array<char32_t, 256> make_byte_to_symbol() {
    std::array<char32_t, 256> table{};
    char32_t next = 0x100;
    for (unsigned b = 0; b < 256; ++b) {
        table[b] = keeps_code_point(b) ? static_cast<char32_t>(b) : next++;
    }
    return table;
}

constexpr std::array<char32_t, 256> kByteToSymbol = make_byte_to_symbol();

// The symbols occupy U+0000-U+0143 (U+0100 + 68 - 1), so the inverse is a
// table indexed by code point; -1 marks a code point that is not a symbol.
constexpr char32_t kSymbolLimit = 0x100 + 68;

constexpr std::array<int, kSymbolLimit> make_symbol_to_byte() {
    std::array<int, kSymbolLimit> table{};
    for (int& entry : table) {
        entry = -1;
    }
    for (unsigned b = 0; b < 256; ++b) {
        table[kByteToSymbol[b]] = static_cast<int>(b);
    }
    return table;
}

constexpr std::array<int, kSymbolLimit> kSymbolToByte = make_symbol_to_byte();

}  // namespace

char32_t byte_to_symbol(std::uint8_t byte) { return kByteToSymbol[byte]; }

std::optional<std::uint8_t> symbol_to_byte(char32_t symbol) {
    if (symbol >= kSymbolLimit || kSymbolToByte[symbol] < 0) {
        return std::nullopt;
    }
    return static_cast<std::uint8_t>(kSymbolToByte[symbol]);
}

std::string bytes_to_symbols(std::string_view bytes) {
    std::string out;
    // Every symbol takes one or two UTF-8 bytes.
    out.reserve(bytes.size() * 2);
    for (const char c : bytes) {
        out += encode_utf8(byte_to_symbol(static_cast<std::uint8_t>(c)));
    }
    return out;
}

std::optional<std::string> symbols_to_bytes(std::string_view symbols) {
    std::string out;
    out.reserve(symbols.size());
    std::size_t pos = 0;
    while (pos < symbols.size()) {
        char32_t cp = 0;
        std::size_t length = 0;
        if (!decode_utf8_at(symbols, pos, cp, length)) {
            return std::nullopt;
        }
        const std::optional<std::uint8_t> byte = symbol_to_byte(cp);
        if (!byte) {
            return std::nullopt;
        }
        out += static_cast<char>(*byte);
        pos += length;
    }
    return out;
}

}  // namespace titan::tokenizer
