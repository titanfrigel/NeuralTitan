#include "io/checkpoint_file.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <system_error>

#include "core/error.hpp"
#include "io/checksum.hpp"

namespace titan::io {

namespace {

static_assert(std::endian::native == std::endian::little,
              "checkpoint tensors are stored as raw little-endian floats");

constexpr std::array<char, 4> kMagic = {'T', 'C', 'K', 'P'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint64_t kPreludeBytes = 28;
constexpr std::uint64_t kMaxHeaderBytes = std::uint64_t{16} << 20;
constexpr std::uint64_t kAlign = 64;
constexpr std::size_t kChunkBytes = std::size_t{1} << 20;
constexpr std::size_t kMaxRank = 4;

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

// Rounds up to a multiple of kAlign. Callers keep `n` far below 2^64.
std::uint64_t align_up(std::uint64_t n) {
    return (n + kAlign - 1) / kAlign * kAlign;
}

std::span<const std::byte> as_bytes(const void* p, std::size_t n) {
    return {static_cast<const std::byte*>(p), n};
}

// One validated entry of the header's "tensors" array.
struct Entry {
    std::string name;
    Shape shape;
    std::uint64_t offset = 0;  // relative to the data section
    std::uint64_t bytes = 0;   // unpadded byte size
    std::uint32_t crc = 0;
};

// The result of the file-level and header checks.
struct Parsed {
    JsonValue meta;
    std::vector<Entry> entries;
    std::uint64_t data_start = 0;  // absolute offset of the data section
};

// Reads exactly `n` bytes or throws.
void read_exact(std::ifstream& in, void* dst, std::size_t n,
                const std::string& ctx, const std::string& what) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    TITAN_CHECK(in.gcount() == static_cast<std::streamsize>(n),
                ctx + "short read of " + what);
}

// Validates one tensor entry against the data section length. Every check
// happens before any allocation.
Entry parse_entry(const JsonValue& v, std::size_t index, std::uint64_t data_len,
                  std::uint64_t expected_offset, const std::string& ctx) {
    const std::string where = "tensors[" + std::to_string(index) + "]";
    TITAN_CHECK(v.is_object(), ctx + where + " is not an object");
    Entry e;
    e.name = v.at("name").as_string();
    TITAN_CHECK(!e.name.empty(), ctx + where + " has an empty name");
    const std::string tctx = ctx + "tensor '" + e.name + "': ";

    const JsonValue::Array& dims = v.at("shape").as_array();
    TITAN_CHECK(!dims.empty() && dims.size() <= kMaxRank,
                tctx + "bad shape: rank " + std::to_string(dims.size()) +
                    ", expected 1 to 4");
    // The largest element count whose byte size fits in 64 bits and in a
    // size_t, so no size computed below can overflow.
    constexpr std::uint64_t kMaxElems =
        std::min<std::uint64_t>(std::numeric_limits<std::uint64_t>::max(),
                                std::numeric_limits<std::size_t>::max()) /
        sizeof(float);
    std::uint64_t count = 1;
    for (const JsonValue& d : dims) {
        const std::int64_t dim = d.as_int();
        TITAN_CHECK(dim >= 1, tctx + "bad shape: dimension " +
                                  std::to_string(dim) + " is not positive");
        TITAN_CHECK(count <= kMaxElems / static_cast<std::uint64_t>(dim),
                    tctx + "bad shape: byte size overflows");
        count *= static_cast<std::uint64_t>(dim);
        e.shape.push_back(dim);
    }
    e.bytes = count * sizeof(float);

    const std::int64_t offset = v.at("offset").as_int();
    TITAN_CHECK(offset >= 0, tctx + "negative offset");
    e.offset = static_cast<std::uint64_t>(offset);
    // Compare by subtraction so offset + bytes cannot overflow.
    TITAN_CHECK(e.offset <= data_len && e.bytes <= data_len - e.offset,
                tctx + "out of bounds: offset " + std::to_string(e.offset) +
                    " plus " + std::to_string(e.bytes) +
                    " bytes exceeds the data section of " +
                    std::to_string(data_len) + " bytes");
    TITAN_CHECK(e.offset == expected_offset,
                tctx + "offset " + std::to_string(e.offset) + ", expected " +
                    std::to_string(expected_offset) +
                    " (tensors are packed in order, 64-byte aligned)");

    const std::int64_t crc = v.at("crc32").as_int();
    TITAN_CHECK(crc >= 0 && crc <= std::numeric_limits<std::uint32_t>::max(),
                tctx + "crc32 out of range");
    e.crc = static_cast<std::uint32_t>(crc);
    return e;
}

// Opens `path`, runs every file-level and header check, and leaves `in` open
// for the caller to read tensor data.
Parsed parse_container(const std::filesystem::path& path, std::ifstream& in) {
    const std::string ctx = "checkpoint '" + path.string() + "': ";

    std::error_code ec;
    const std::uintmax_t file_size = std::filesystem::file_size(path, ec);
    TITAN_CHECK(!ec, ctx + "cannot stat: " + ec.message());
    TITAN_CHECK(file_size >= kPreludeBytes,
                ctx + "file size " + std::to_string(file_size) +
                    " is shorter than the 28-byte prelude");

    in.open(path, std::ios::binary);
    TITAN_CHECK(in.is_open(), ctx + "cannot open");
    std::array<unsigned char, kPreludeBytes> prelude{};
    read_exact(in, prelude.data(), prelude.size(), ctx, "the prelude");

    for (std::size_t i = 0; i < kMagic.size(); ++i) {
        TITAN_CHECK(prelude[i] == static_cast<unsigned char>(kMagic[i]),
                    ctx + "bad magic, expected \"TCKP\"");
    }
    const std::uint32_t version = get_u32(prelude.data() + 4);
    TITAN_CHECK(version == kVersion,
                ctx + "unsupported version " + std::to_string(version) +
                    ", expected " + std::to_string(kVersion));
    const std::uint64_t total = get_u64(prelude.data() + 8);
    TITAN_CHECK(total == file_size, ctx + "total size mismatch: header says " +
                                        std::to_string(total) +
                                        " bytes, the file is " +
                                        std::to_string(file_size) + " bytes");
    const std::uint64_t header_len = get_u64(prelude.data() + 16);
    TITAN_CHECK(header_len >= 1 && header_len <= kMaxHeaderBytes,
                ctx + "header length " + std::to_string(header_len) +
                    " is outside 1 to " + std::to_string(kMaxHeaderBytes) +
                    " bytes");
    TITAN_CHECK(header_len <= file_size - kPreludeBytes,
                ctx + "header length " + std::to_string(header_len) +
                    " runs past the end of the file");
    const std::uint32_t header_crc = get_u32(prelude.data() + 24);

    std::string header(static_cast<std::size_t>(header_len), '\0');
    read_exact(in, header.data(), header.size(), ctx, "the header");
    TITAN_CHECK(crc32(as_bytes(header.data(), header.size())) == header_crc,
                ctx + "header checksum mismatch");

    Parsed parsed;
    parsed.data_start = align_up(kPreludeBytes + header_len);
    TITAN_CHECK(parsed.data_start <= file_size,
                ctx + "file ends before the data section");
    const std::uint64_t data_len = file_size - parsed.data_start;

    JsonValue root;
    try {
        root = parse_json(header);
    } catch (const Error& err) {
        throw Error(ctx + "header is not valid JSON: " + err.what());
    }
    TITAN_CHECK(root.is_object(), ctx + "header is not a JSON object");
    const JsonValue* meta = root.find("meta");
    TITAN_CHECK(meta != nullptr && meta->is_object(),
                ctx + "header has no 'meta' object");
    parsed.meta = *meta;
    const JsonValue* list = root.find("tensors");
    TITAN_CHECK(list != nullptr && list->is_array(),
                ctx + "header has no 'tensors' array");

    std::set<std::string> seen;
    std::uint64_t cursor = 0;
    for (std::size_t i = 0; i < list->as_array().size(); ++i) {
        Entry e = parse_entry(list->as_array()[i], i, data_len, cursor, ctx);
        TITAN_CHECK(seen.insert(e.name).second,
                    ctx + "duplicate tensor name '" + e.name + "'");
        cursor = align_up(e.offset + e.bytes);
        parsed.entries.push_back(std::move(e));
    }
    TITAN_CHECK(cursor == data_len,
                ctx + "data section is " + std::to_string(data_len) +
                    " bytes but the tensors fill " + std::to_string(cursor));
    return parsed;
}

}  // namespace

void write_checkpoint_file(const std::filesystem::path& path,
                           const JsonValue& meta,
                           std::span<const NamedTensor> tensors) {
    const std::string ctx = "write_checkpoint_file: '" + path.string() + "': ";
    TITAN_CHECK(meta.is_object(), ctx + "meta must be a JSON object");

    // Validate everything and lay out the data section before touching disk.
    std::set<std::string> seen;
    JsonValue list = JsonValue::array();
    std::uint64_t cursor = 0;
    for (const NamedTensor& t : tensors) {
        TITAN_CHECK(!t.name.empty(), ctx + "tensor with an empty name");
        TITAN_CHECK(seen.insert(t.name).second,
                    ctx + "duplicate tensor name '" + t.name + "'");
        TITAN_CHECK(t.value.device() == Device::CPU,
                    ctx + "tensor '" + t.name + "' is not on the CPU");
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(t.value.numel()) * sizeof(float);
        const std::uint32_t crc =
            crc32(as_bytes(t.value.data(), static_cast<std::size_t>(bytes)));

        JsonValue shape = JsonValue::array();
        for (std::int64_t d : t.value.shape()) {
            shape.push_back(JsonValue::number(static_cast<double>(d)));
        }
        JsonValue entry = JsonValue::object();
        entry.set("name", JsonValue::string(t.name));
        entry.set("shape", std::move(shape));
        entry.set("offset", JsonValue::number(static_cast<double>(cursor)));
        entry.set("crc32", JsonValue::number(static_cast<double>(crc)));
        list.push_back(std::move(entry));
        cursor += align_up(bytes);
    }
    JsonValue root = JsonValue::object();
    root.set("meta", meta);
    root.set("tensors", std::move(list));
    const std::string header = to_json(root, 0);
    TITAN_CHECK(header.size() <= kMaxHeaderBytes,
                ctx + "header of " + std::to_string(header.size()) +
                    " bytes exceeds the 16 MiB limit");

    const std::uint64_t data_start = align_up(kPreludeBytes + header.size());
    const std::uint64_t total = data_start + cursor;

    std::array<unsigned char, kPreludeBytes> prelude{};
    for (std::size_t i = 0; i < kMagic.size(); ++i) {
        prelude[i] = static_cast<unsigned char>(kMagic[i]);
    }
    put_u32(prelude.data() + 4, kVersion);
    put_u64(prelude.data() + 8, total);
    put_u64(prelude.data() + 16, header.size());
    put_u32(prelude.data() + 24, crc32(as_bytes(header.data(), header.size())));

    std::filesystem::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        TITAN_CHECK(out.is_open(), ctx + "cannot open '" + tmp.string() + "'");
        const std::array<char, kAlign> zeros{};
        const auto put = [&](const void* p, std::uint64_t n) {
            out.write(static_cast<const char*>(p),
                      static_cast<std::streamsize>(n));
        };
        put(prelude.data(), prelude.size());
        put(header.data(), header.size());
        put(zeros.data(), data_start - kPreludeBytes - header.size());
        for (const NamedTensor& t : tensors) {
            const std::uint64_t bytes =
                static_cast<std::uint64_t>(t.value.numel()) * sizeof(float);
            put(t.value.data(), bytes);
            put(zeros.data(), align_up(bytes) - bytes);
        }
        out.flush();
        if (!out.good()) {
            out.close();
            std::error_code ignored;
            std::filesystem::remove(tmp, ignored);
            TITAN_CHECK(false, ctx + "failed writing '" + tmp.string() + "'");
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        TITAN_CHECK(false, ctx + "cannot rename '" + tmp.string() +
                               "' into place: " + ec.message());
    }
}

CheckpointFile read_checkpoint_file(const std::filesystem::path& path) {
    std::ifstream in;
    Parsed parsed = parse_container(path, in);
    const std::string ctx = "checkpoint '" + path.string() + "': ";

    CheckpointFile result;
    result.meta = std::move(parsed.meta);
    for (const Entry& e : parsed.entries) {
        const std::string tctx = ctx + "tensor '" + e.name + "': ";
        // The entry was bounds-checked against the file size, so this
        // allocation is no larger than the file.
        Tensor value = Tensor::empty(e.shape, Device::CPU);
        in.clear();
        in.seekg(static_cast<std::streamoff>(parsed.data_start + e.offset));
        TITAN_CHECK(in.good(), tctx + "cannot seek to its data");

        auto* dst = reinterpret_cast<std::byte*>(value.data());
        std::uint32_t crc = 0;
        for (std::uint64_t done = 0; done < e.bytes;) {
            const std::size_t n = static_cast<std::size_t>(
                std::min<std::uint64_t>(kChunkBytes, e.bytes - done));
            read_exact(in, dst + done, n, tctx, "its data");
            crc = crc32(std::span<const std::byte>(dst + done, n), crc);
            done += n;
        }
        TITAN_CHECK(crc == e.crc, tctx + "checksum mismatch");
        result.tensors.push_back({e.name, std::move(value)});
    }
    return result;
}

JsonValue read_checkpoint_meta(const std::filesystem::path& path) {
    std::ifstream in;
    return parse_container(path, in).meta;
}

}  // namespace titan::io
