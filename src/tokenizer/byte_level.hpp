#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace titan::tokenizer {

// GPT-2's byte-level alphabet (`bytes_to_unicode`, docs/math/bpe.md,
// "Byte-level alphabet"): a one-to-one map from the 256 byte values to 256
// printable code points. Bytes 33-126, 161-172 and 174-255 map to the same
// code point; the other 68 bytes, in increasing order, map to U+0100, U+0101,
// ... So the space 0x20 is U+0120 'Ġ' and the newline 0x0A is U+010A 'Ċ'.
char32_t byte_to_symbol(std::uint8_t byte);

// The inverse of byte_to_symbol, or nullopt when `symbol` is not one of the
// 256 symbols.
std::optional<std::uint8_t> symbol_to_byte(char32_t symbol);

// Maps every byte of `bytes` to its symbol and returns the symbols as UTF-8
// text. Any byte string is accepted, including invalid UTF-8.
std::string bytes_to_symbols(std::string_view bytes);

// The inverse of bytes_to_symbols: the bytes of UTF-8 text written in the
// byte-level alphabet, or nullopt when `symbols` is invalid UTF-8 or holds a
// code point outside the alphabet.
std::optional<std::string> symbols_to_bytes(std::string_view symbols);

}  // namespace titan::tokenizer
