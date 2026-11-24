#include "io/checksum.hpp"

#include <array>

namespace titan::io {

namespace {

// Table entry n is the CRC register after feeding the single byte n through
// eight shift steps of the reflected polynomial.
constexpr std::array<std::uint32_t, 256> make_table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t n = 0; n < 256; ++n) {
        std::uint32_t c = n;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1U) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
        }
        table[n] = c;
    }
    return table;
}

constexpr std::array<std::uint32_t, 256> kTable = make_table();

}  // namespace

std::uint32_t crc32(std::span<const std::byte> data, std::uint32_t seed) {
    // The seed is a finished CRC; undo the final XOR to resume the register.
    std::uint32_t c = ~seed;
    for (std::byte b : data) {
        c = kTable[(c ^ static_cast<std::uint32_t>(b)) & 0xFFU] ^ (c >> 8);
    }
    return ~c;
}

}  // namespace titan::io
