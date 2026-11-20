#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

namespace titan::io {

// Reads a whole file. Throws titan::Error naming the path if the file is
// missing, unreadable or larger than `max_bytes`.
std::string read_file(const std::filesystem::path& path,
                      std::size_t max_bytes = std::size_t{1} << 30);

// Writes `content` to `path`, replacing any existing file. Throws titan::Error
// naming the path on failure.
void write_file(const std::filesystem::path& path, std::string_view content);

}  // namespace titan::io
