#include "io/file.hpp"

#include <fstream>
#include <system_error>

#include "core/error.hpp"

namespace titan::io {

std::string read_file(const std::filesystem::path& path,
                      std::size_t max_bytes) {
    const std::string name = path.string();
    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    TITAN_CHECK(!ec, "read_file: cannot stat '" + name + "': " + ec.message());
    TITAN_CHECK(size <= max_bytes,
                "read_file: '" + name + "' is " + std::to_string(size) +
                    " bytes, over the limit of " + std::to_string(max_bytes));

    std::ifstream in(path, std::ios::binary);
    TITAN_CHECK(in.is_open(), "read_file: cannot open '" + name + "'");
    std::string content(static_cast<std::size_t>(size), '\0');
    in.read(content.data(), static_cast<std::streamsize>(content.size()));
    TITAN_CHECK(static_cast<std::uintmax_t>(in.gcount()) == size,
                "read_file: short read of '" + name + "'");
    return content;
}

void write_file(const std::filesystem::path& path, std::string_view content) {
    const std::string name = path.string();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    TITAN_CHECK(out.is_open(), "write_file: cannot open '" + name + "'");
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.flush();
    TITAN_CHECK(out.good(), "write_file: failed writing '" + name + "'");
}

}  // namespace titan::io
