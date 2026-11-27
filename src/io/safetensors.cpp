#include "io/safetensors.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <fstream>
#include <limits>
#include <system_error>
#include <utility>

#include "core/error.hpp"
#include "io/json_fields.hpp"

namespace titan::io {

namespace {

static_assert(std::endian::native == std::endian::little,
              "safetensors data is little-endian and is read in place");

constexpr std::uint64_t kLengthBytes = 8;
constexpr std::string_view kMetadataKey = "__metadata__";

std::uint64_t dtype_size(SafetensorsDtype dtype) {
    return dtype == SafetensorsDtype::F32 ? 4 : 2;
}

SafetensorsDtype parse_dtype(const std::string& s) {
    if (s == "F32") return SafetensorsDtype::F32;
    if (s == "F16") return SafetensorsDtype::F16;
    return SafetensorsDtype::BF16;  // JsonFields::choice allowed nothing else
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        s += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    }
    return s + "]";
}

// Validates one tensor entry against the data section length `data_len`.
// Contiguity with the other entries is checked by the caller.
SafetensorsEntry parse_entry(const std::string& name, const JsonValue& v,
                             std::uint64_t data_len,
                             const std::string& context) {
    const JsonFields fields(v, context, name,
                            {"dtype", "shape", "data_offsets"});
    SafetensorsEntry e;
    e.name = name;
    const std::string dtype =
        fields.choice(fields.require("dtype"), "dtype", {"F32", "F16", "BF16"});
    e.dtype = parse_dtype(dtype);

    const JsonValue& shape = fields.require("shape");
    if (!shape.is_array()) {
        fields.fail(
            fields.path_of("shape"),
            std::string("expected an array, got ") + json_kind_name(shape));
    }
    // The element count as int64, the type of Shape; a dimension of 0 makes
    // the tensor empty, and the product is checked before each multiply.
    std::int64_t count = 1;
    for (const JsonValue& d : shape.as_array()) {
        const std::int64_t dim = fields.integer(d, "shape", 0);
        if (dim != 0 &&
            count > std::numeric_limits<std::int64_t>::max() / dim) {
            fields.fail(fields.path_of("shape"),
                        "element count overflows int64");
        }
        count *= dim;
        e.shape.push_back(dim);
    }

    const JsonValue& offsets = fields.require("data_offsets");
    const std::string offsets_path = fields.path_of("data_offsets");
    if (!offsets.is_array() || offsets.as_array().size() != 2) {
        fields.fail(offsets_path, "expected an array [begin, end]");
    }
    // integer() bounds both by 2^53, so the absolute offset data_start + begin
    // that read_f32 seeks to cannot overflow.
    e.begin = static_cast<std::uint64_t>(
        fields.integer(offsets.as_array()[0], "data_offsets", 0));
    e.end = static_cast<std::uint64_t>(
        fields.integer(offsets.as_array()[1], "data_offsets", 0));
    if (e.end < e.begin) {
        fields.fail(offsets_path, "end " + std::to_string(e.end) +
                                      " is before begin " +
                                      std::to_string(e.begin));
    }
    if (e.end > data_len) {
        fields.fail(offsets_path, "end " + std::to_string(e.end) +
                                      " is past the data section of " +
                                      std::to_string(data_len) + " bytes");
    }
    // count < 2^63 and size <= 4, so count · size can overflow 64 bits; such
    // a count needs more bytes than any file holds.
    const std::uint64_t bytes = e.end - e.begin;
    const std::uint64_t size = dtype_size(e.dtype);
    const auto ucount = static_cast<std::uint64_t>(count);
    const bool overflows =
        ucount > std::numeric_limits<std::uint64_t>::max() / size;
    if (overflows || ucount * size != bytes) {
        const std::string needed =
            overflows ? "more than 2^64" : std::to_string(ucount * size);
        fields.fail(offsets_path, "the range holds " + std::to_string(bytes) +
                                      " bytes, but shape " +
                                      shape_string(e.shape) + " of " + dtype +
                                      " needs " + needed);
    }
    return e;
}

// Checks that the entries, sorted by begin, tile [0, data_len) exactly: no
// gap, no overlap, nothing left over. Sorts `entries` by (begin, end), which
// puts an empty tensor before a non-empty one starting at the same offset.
void check_contiguous(std::vector<SafetensorsEntry>& entries,
                      std::uint64_t data_len, const std::string& context) {
    std::sort(entries.begin(), entries.end(),
              [](const SafetensorsEntry& a, const SafetensorsEntry& b) {
                  return std::pair(a.begin, a.end) < std::pair(b.begin, b.end);
              });
    std::uint64_t expected = 0;
    for (const SafetensorsEntry& e : entries) {
        TITAN_CHECK(e.begin == expected,
                    context + ": data not contiguous: tensor '" + e.name +
                        "' begins at " + std::to_string(e.begin) +
                        ", expected " + std::to_string(expected));
        expected = e.end;
    }
    const std::string last =
        entries.empty() ? "no tensor" : "tensor '" + entries.back().name + "'";
    TITAN_CHECK(expected == data_len,
                context + ": data not contiguous: the last range (" + last +
                    ") ends at " + std::to_string(expected) +
                    ", but the data section has " + std::to_string(data_len) +
                    " bytes");
}

}  // namespace

SafetensorsFile SafetensorsFile::open(const std::filesystem::path& path) {
    const std::string context = "safetensors '" + path.string() + "'";

    std::error_code ec;
    const std::uintmax_t file_size = std::filesystem::file_size(path, ec);
    TITAN_CHECK(!ec, context + ": cannot stat: " + ec.message());
    TITAN_CHECK(file_size >= kLengthBytes,
                context + ": file size " + std::to_string(file_size) +
                    " is shorter than the 8-byte header length");

    std::ifstream in(path, std::ios::binary);
    TITAN_CHECK(in.is_open(), context + ": cannot open");
    std::array<unsigned char, kLengthBytes> prefix{};
    in.read(reinterpret_cast<char*>(prefix.data()), prefix.size());
    TITAN_CHECK(in.gcount() == static_cast<std::streamsize>(prefix.size()),
                context + ": short read of the header length");
    std::uint64_t header_len = 0;
    for (std::size_t i = 0; i < kLengthBytes; ++i) {
        header_len |= static_cast<std::uint64_t>(prefix[i]) << (8 * i);
    }
    TITAN_CHECK(header_len <= kMaxSafetensorsHeaderBytes,
                context + ": header length " + std::to_string(header_len) +
                    " is over the limit of " +
                    std::to_string(kMaxSafetensorsHeaderBytes) + " bytes");
    TITAN_CHECK(header_len <= file_size - kLengthBytes,
                context + ": header length " + std::to_string(header_len) +
                    " runs past the end of the file of " +
                    std::to_string(file_size) + " bytes");

    // The one bounded read of the header; tensor data is read on demand.
    std::string header(static_cast<std::size_t>(header_len), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    TITAN_CHECK(in.gcount() == static_cast<std::streamsize>(header.size()),
                context + ": short read of the header");

    // The JSON parser accepts trailing whitespace, which covers the padding
    // spaces, and rejects any other trailing byte and duplicate keys.
    JsonValue root;
    try {
        root = parse_json(header);
    } catch (const Error& err) {
        throw_error(__FILE__, __LINE__,
                    context + ": header is not valid JSON: " + err.what());
    }
    TITAN_CHECK(root.is_object(), context + ": header is not a JSON object");

    SafetensorsFile f;
    f.path_ = path;
    f.data_start_ = kLengthBytes + header_len;
    f.metadata_ = JsonValue::object();
    const std::uint64_t data_len = file_size - f.data_start_;

    for (const auto& [key, value] : root.as_object()) {
        if (key == kMetadataKey) {
            TITAN_CHECK(value.is_object(), context + ": " + key +
                                               ": expected an object, got " +
                                               json_kind_name(value));
            for (const auto& [mkey, mvalue] : value.as_object()) {
                TITAN_CHECK(mvalue.is_string(),
                            context + ": " + key + "." + mkey +
                                ": expected a string, got " +
                                json_kind_name(mvalue));
            }
            f.metadata_ = value;
        } else {
            f.entries_.push_back(parse_entry(key, value, data_len, context));
        }
    }

    check_contiguous(f.entries_, data_len, context);
    std::sort(f.entries_.begin(), f.entries_.end(),
              [](const SafetensorsEntry& a, const SafetensorsEntry& b) {
                  return a.name < b.name;
              });
    return f;
}

const SafetensorsEntry* SafetensorsFile::find(std::string_view name) const {
    const auto it =
        std::lower_bound(entries_.begin(), entries_.end(), name,
                         [](const SafetensorsEntry& e, std::string_view n) {
                             return e.name < n;
                         });
    return it != entries_.end() && it->name == name ? &*it : nullptr;
}

std::vector<float> SafetensorsFile::read_f32(
    const SafetensorsEntry& entry) const {
    const std::string context =
        "safetensors '" + path_.string() + "': tensor '" + entry.name + "'";
    // open() checked end - begin == count · size, and that the range lies in
    // the data section.
    const auto bytes = static_cast<std::size_t>(entry.end - entry.begin);
    const std::size_t count = bytes / dtype_size(entry.dtype);
    std::vector<float> out(count);
    if (count == 0) return out;

    std::ifstream in(path_, std::ios::binary);
    TITAN_CHECK(in.is_open(), context + ": cannot open the file");
    in.seekg(static_cast<std::streamoff>(data_start_ + entry.begin));
    if (entry.dtype == SafetensorsDtype::F32) {
        // Little-endian FP32 is already the in-memory layout.
        in.read(reinterpret_cast<char*>(out.data()),
                static_cast<std::streamsize>(bytes));
        TITAN_CHECK(
            in.gcount() == static_cast<std::streamsize>(bytes),
            context + ": short read of " + std::to_string(bytes) + " bytes");
        return out;
    }

    std::vector<std::uint16_t> raw(count);
    in.read(reinterpret_cast<char*>(raw.data()),
            static_cast<std::streamsize>(bytes));
    TITAN_CHECK(
        in.gcount() == static_cast<std::streamsize>(bytes),
        context + ": short read of " + std::to_string(bytes) + " bytes");
    const auto convert =
        entry.dtype == SafetensorsDtype::BF16 ? bf16_to_f32 : f16_to_f32;
    std::transform(raw.begin(), raw.end(), out.begin(), convert);
    return out;
}

// BF16 is the top half of an FP32: 1 sign bit, the same 8 exponent bits and
// the 7 leading mantissa bits. Shifting it into the high half and zeroing the
// 16 low mantissa bits gives the FP32 with exactly that value, including
// zeros, subnormals, infinities and NaNs.
float bf16_to_f32(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

// F16: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits. FP32 has 8
// exponent bits (bias 127) and 23 mantissa bits, so every F16 value is exactly
// representable; the exponent field picks the case.
float f16_to_f32(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1FU;
    std::uint32_t mantissa = bits & 0x3FFU;

    if (exponent == 0x1F) {
        // Infinity (mantissa 0) or NaN: all-ones exponent, mantissa (and so
        // the NaN payload) moved to the top of the 23-bit field.
        return std::bit_cast<float>(sign | 0x7F800000U | (mantissa << 13));
    }
    if (exponent == 0) {
        if (mantissa == 0) return std::bit_cast<float>(sign);  // ±0
        // Subnormal: value = mantissa · 2^-24 = 0.mantissa · 2^-14. FP32
        // stores it as a normal number: shift the mantissa left until its
        // leading 1 reaches bit 10, the implicit-1 position, lowering the
        // exponent by one per shift, then drop that implicit bit.
        std::uint32_t e = 127 - 14;
        while ((mantissa & 0x400U) == 0) {
            mantissa <<= 1;
            --e;
        }
        mantissa &= 0x3FFU;
        return std::bit_cast<float>(sign | (e << 23) | (mantissa << 13));
    }
    // Normal: rebias the exponent, 2^(exponent - 15) = 2^((exponent + 112) -
    // 127), and widen the mantissa from 10 to 23 bits.
    return std::bit_cast<float>(sign | ((exponent + 112) << 23) |
                                (mantissa << 13));
}

}  // namespace titan::io
