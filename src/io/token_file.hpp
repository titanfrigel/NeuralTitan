#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace titan::io {

// Token file layout, all integers little-endian:
//   bytes 0..3   ASCII magic "TTOK"
//   bytes 4..7   u32 version, currently 1
//   bytes 8..15  u64 count
//   then         `count` int32 token ids
// The file size must be exactly 16 + 4 * count.
void write_token_file(const std::filesystem::path& path,
                      std::span<const std::int32_t> ids);

// Throws titan::Error naming the path on a missing file, a wrong magic or
// version, or a size that does not match the count.
std::vector<std::int32_t> read_token_file(const std::filesystem::path& path);

}  // namespace titan::io
