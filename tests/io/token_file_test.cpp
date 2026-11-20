#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "harness/test.hpp"
#include "io/file.hpp"
#include "io/token_file.hpp"

namespace {

std::filesystem::path temp_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

// A valid header followed by `payload_bytes` zero bytes.
std::string file_bytes(const std::string& magic, unsigned char version,
                       unsigned char count, std::size_t payload_bytes) {
    std::string s = magic;
    s += std::string{static_cast<char>(version), 0, 0, 0};
    s += std::string{static_cast<char>(count), 0, 0, 0, 0, 0, 0, 0};
    s += std::string(payload_bytes, '\0');
    return s;
}

bool read_fails_naming(const std::filesystem::path& p,
                       const std::string& content) {
    titan::io::write_file(p, content);
    bool ok = false;
    try {
        titan::io::read_token_file(p);
    } catch (const titan::Error& e) {
        ok = std::string(e.what()).find(p.filename().string()) !=
             std::string::npos;
    }
    std::filesystem::remove(p);
    return ok;
}

}  // namespace

TITAN_TEST(TokenFile, RoundTrip) {
    const auto p = temp_path("titan_token_roundtrip.bin");
    std::vector<std::int32_t> ids = {0,    1, -1, 2147483647, -2147483647 - 1,
                                     65537};
    for (int i = 0; i < 70000; ++i) ids.push_back(i * 3);  // crosses chunks
    titan::io::write_token_file(p, ids);
    TITAN_EXPECT_EQ(std::filesystem::file_size(p), 16 + 4 * ids.size());
    TITAN_EXPECT(titan::io::read_token_file(p) == ids);

    titan::io::write_token_file(p, {});
    TITAN_EXPECT(titan::io::read_token_file(p).empty());
    std::filesystem::remove(p);
}

TITAN_TEST(TokenFile, LayoutIsLittleEndian) {
    const auto p = temp_path("titan_token_layout.bin");
    const std::vector<std::int32_t> ids = {0x01020304};
    titan::io::write_token_file(p, ids);
    TITAN_EXPECT_EQ(
        titan::io::read_file(p),
        file_bytes("TTOK", 1, 1, 0) + std::string("\x04\x03\x02\x01", 4));
    std::filesystem::remove(p);
}

TITAN_TEST(TokenFile, RejectsMalformed) {
    const auto p = temp_path("titan_token_bad.bin");
    TITAN_EXPECT(read_fails_naming(p, file_bytes("TTOX", 1, 1, 4)));  // magic
    TITAN_EXPECT(read_fails_naming(p, file_bytes("TTOK", 2, 1, 4)));  // version
    TITAN_EXPECT(
        read_fails_naming(p, file_bytes("TTOK", 1, 2, 4)));  // truncated
    TITAN_EXPECT(
        read_fails_naming(p, file_bytes("TTOK", 1, 1, 8)));  // trailing
    TITAN_EXPECT(read_fails_naming(p, file_bytes("TTOK", 1, 1, 5)));  // ragged
    TITAN_EXPECT(
        read_fails_naming(p, std::string("TTOK\x01\0\0", 7)));  // short header
    std::string huge = file_bytes("TTOK", 1, 0, 0);
    huge.replace(8, 8, std::string(8, '\xff'));  // count near 2^64
    TITAN_EXPECT(read_fails_naming(p, huge));
    TITAN_EXPECT_THROWS(
        titan::io::read_token_file(temp_path("titan_no_such_tokens.bin")),
        titan::Error);
}
