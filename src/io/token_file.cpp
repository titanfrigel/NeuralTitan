#include "io/token_file.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>

#include "core/error.hpp"

namespace titan::io {

namespace {

constexpr std::array<char, 4> kMagic = {'T', 'T', 'O', 'K'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint64_t kHeaderBytes = 16;
constexpr std::uint64_t kBytesPerId = 4;
// Ids are converted this many at a time so a large file is not held twice.
constexpr std::size_t kChunkIds = 1 << 16;

void put_u32(unsigned char* out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<unsigned char>(v >> (8 * i));
    }
}

void put_u64(unsigned char* out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<unsigned char>(v >> (8 * i));
    }
}

std::uint32_t get_u32(const unsigned char* in) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<std::uint32_t>(in[i]) << (8 * i);
    }
    return v;
}

std::uint64_t get_u64(const unsigned char* in) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(in[i]) << (8 * i);
    }
    return v;
}

}  // namespace

void write_token_file(const std::filesystem::path& path,
                      std::span<const std::int32_t> ids) {
    const std::string name = path.string();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    TITAN_CHECK(out.is_open(), "write_token_file: cannot open '" + name + "'");

    std::array<unsigned char, kHeaderBytes> header{};
    for (std::size_t i = 0; i < kMagic.size(); ++i) {
        header[i] = static_cast<unsigned char>(kMagic[i]);
    }
    put_u32(header.data() + 4, kVersion);
    put_u64(header.data() + 8, static_cast<std::uint64_t>(ids.size()));
    out.write(reinterpret_cast<const char*>(header.data()),
              static_cast<std::streamsize>(header.size()));

    std::vector<unsigned char> buf;
    for (std::size_t start = 0; start < ids.size(); start += kChunkIds) {
        const std::size_t n = std::min(kChunkIds, ids.size() - start);
        buf.assign(n * kBytesPerId, 0);
        for (std::size_t i = 0; i < n; ++i) {
            put_u32(buf.data() + i * kBytesPerId,
                    static_cast<std::uint32_t>(ids[start + i]));
        }
        out.write(reinterpret_cast<const char*>(buf.data()),
                  static_cast<std::streamsize>(buf.size()));
    }
    out.flush();
    TITAN_CHECK(out.good(), "write_token_file: failed writing '" + name + "'");
}

std::vector<std::int32_t> read_token_file(const std::filesystem::path& path) {
    const std::string ctx = "read_token_file: '" + path.string() + "': ";

    std::error_code ec;
    const std::uintmax_t file_size = std::filesystem::file_size(path, ec);
    TITAN_CHECK(!ec, ctx + "cannot stat: " + ec.message());
    TITAN_CHECK(file_size >= kHeaderBytes,
                ctx + "file is " + std::to_string(file_size) +
                    " bytes, shorter than the 16-byte header");

    std::ifstream in(path, std::ios::binary);
    TITAN_CHECK(in.is_open(), ctx + "cannot open");
    std::array<unsigned char, kHeaderBytes> header{};
    in.read(reinterpret_cast<char*>(header.data()),
            static_cast<std::streamsize>(header.size()));
    TITAN_CHECK(in.gcount() == static_cast<std::streamsize>(header.size()),
                ctx + "short read of the header");

    for (std::size_t i = 0; i < kMagic.size(); ++i) {
        TITAN_CHECK(header[i] == static_cast<unsigned char>(kMagic[i]),
                    ctx + "bad magic, expected \"TTOK\"");
    }
    const std::uint32_t version = get_u32(header.data() + 4);
    TITAN_CHECK(version == kVersion,
                ctx + "unsupported version " + std::to_string(version) +
                    ", expected " + std::to_string(kVersion));

    // Compare count with payload / 4 instead of multiplying, so a huge count
    // cannot overflow.
    const std::uint64_t count = get_u64(header.data() + 8);
    const std::uint64_t payload = file_size - kHeaderBytes;
    TITAN_CHECK(payload % kBytesPerId == 0 && payload / kBytesPerId == count,
                ctx + "header says " + std::to_string(count) +
                    " ids but the payload is " + std::to_string(payload) +
                    " bytes");
    TITAN_CHECK(count <= std::numeric_limits<std::size_t>::max(),
                ctx + "too many ids for this platform");

    std::vector<std::int32_t> ids(static_cast<std::size_t>(count));
    std::vector<unsigned char> buf;
    for (std::size_t start = 0; start < ids.size(); start += kChunkIds) {
        const std::size_t n = std::min(kChunkIds, ids.size() - start);
        buf.resize(n * kBytesPerId);
        in.read(reinterpret_cast<char*>(buf.data()),
                static_cast<std::streamsize>(buf.size()));
        TITAN_CHECK(in.gcount() == static_cast<std::streamsize>(buf.size()),
                    ctx + "short read of the payload");
        for (std::size_t i = 0; i < n; ++i) {
            ids[start + i] = static_cast<std::int32_t>(
                get_u32(buf.data() + i * kBytesPerId));
        }
    }
    return ids;
}

}  // namespace titan::io
