#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace titan::io {

// CRC-32/ISO-HDLC (the zlib CRC): reflected polynomial 0xEDB88320, initial
// value and final XOR 0xFFFFFFFF. Pass the previous result as `seed` to
// checksum data in pieces: crc32(b, crc32(a)) == crc32(a followed by b).
std::uint32_t crc32(std::span<const std::byte> data, std::uint32_t seed = 0);

}  // namespace titan::io
