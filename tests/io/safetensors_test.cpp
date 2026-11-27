#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "harness/test.hpp"
#include "io/file.hpp"
#include "io/safetensors.hpp"
#include "support/safetensors_writer.hpp"

namespace {

using titan::io::SafetensorsDtype;
using titan::io::SafetensorsFile;
using titan::test::TestTensor;
using titan::test::write_safetensors;

std::filesystem::path temp_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

// Raw little-endian bytes of FP32 values or of 16-bit patterns.
std::string f32_bytes(std::initializer_list<float> values) {
    std::string out;
    for (float v : values) {
        const auto bits = std::bit_cast<std::uint32_t>(v);
        for (int i = 0; i < 4; ++i) {
            out += static_cast<char>((bits >> (8 * i)) & 0xFF);
        }
    }
    return out;
}

std::string u16_bytes(std::initializer_list<std::uint16_t> values) {
    std::string out;
    for (std::uint16_t v : values) {
        out += static_cast<char>(v & 0xFF);
        out += static_cast<char>(v >> 8);
    }
    return out;
}

bool is_negative_zero(float v) {
    return v == 0.0F && std::bit_cast<std::uint32_t>(v) == 0x80000000U;
}

// Opens `path` and returns whether it threw a titan::Error whose message holds
// the file name and every one of `needles`.
bool open_rejects(const std::filesystem::path& path,
                  std::initializer_list<const char*> needles) {
    try {
        SafetensorsFile::open(path);
    } catch (const titan::Error& e) {
        const std::string message = e.what();
        bool ok = message.find(path.filename().string()) != std::string::npos;
        for (const char* n : needles) {
            ok = ok && message.find(n) != std::string::npos;
        }
        return ok;
    }
    return false;
}

// Writes `header` verbatim followed by `data_bytes` zero bytes and checks that
// open() rejects it with every one of `needles` in the message.
bool header_rejects(const char* file, const std::string& header,
                    std::size_t data_bytes,
                    std::initializer_list<const char*> needles) {
    const auto p = temp_path(file);
    write_safetensors(p, {{"data", "U8", {}, std::string(data_bytes, '\0')}},
                      header);
    const bool ok = open_rejects(p, needles);
    std::filesystem::remove(p);
    return ok;
}

bool bytes_reject(const char* file, const std::string& bytes,
                  std::initializer_list<const char*> needles) {
    const auto p = temp_path(file);
    titan::io::write_file(p, bytes);
    const bool ok = open_rejects(p, needles);
    std::filesystem::remove(p);
    return ok;
}

}  // namespace

TITAN_TEST(Safetensors, ReadsAllDtypes) {
    const auto p = temp_path("titan_st_dtypes.safetensors");
    // Written out of name order, to check that entries() sorts.
    write_safetensors(
        p,
        {{"z.f32", "F32", {2, 3}, f32_bytes({1, -2, 0.5F, 3, 1e-3F, -7})},
         {"m.f16", "F16", {3}, u16_bytes({0x3C00, 0xC000, 0x3800})},
         {"a.bf16", "BF16", {2, 2}, u16_bytes({0x3F80, 0xC040, 0x4000, 0})}});
    const SafetensorsFile f = SafetensorsFile::open(p);

    const auto& entries = f.entries();
    TITAN_EXPECT_EQ(entries.size(), std::size_t{3});
    TITAN_EXPECT_EQ(entries[0].name, std::string("a.bf16"));
    TITAN_EXPECT_EQ(entries[1].name, std::string("m.f16"));
    TITAN_EXPECT_EQ(entries[2].name, std::string("z.f32"));
    TITAN_EXPECT(f.metadata().is_object());
    TITAN_EXPECT(f.metadata().as_object().empty());
    TITAN_EXPECT(f.find("missing") == nullptr);

    const auto* f32 = f.find("z.f32");
    TITAN_EXPECT(f32 != nullptr);
    TITAN_EXPECT(f32->dtype == SafetensorsDtype::F32);
    TITAN_EXPECT(f32->shape == (std::vector<std::int64_t>{2, 3}));
    TITAN_EXPECT_EQ(f32->begin, std::uint64_t{0});
    TITAN_EXPECT_EQ(f32->end, std::uint64_t{24});
    TITAN_EXPECT(f.read_f32(*f32) ==
                 (std::vector<float>{1, -2, 0.5F, 3, 1e-3F, -7}));

    const auto* f16 = f.find("m.f16");
    TITAN_EXPECT(f16 != nullptr);
    TITAN_EXPECT(f16->dtype == SafetensorsDtype::F16);
    TITAN_EXPECT(f16->shape == (std::vector<std::int64_t>{3}));
    TITAN_EXPECT(f.read_f32(*f16) == (std::vector<float>{1, -2, 0.5F}));

    const auto* bf16 = f.find("a.bf16");
    TITAN_EXPECT(bf16 != nullptr);
    TITAN_EXPECT(bf16->dtype == SafetensorsDtype::BF16);
    TITAN_EXPECT(bf16->shape == (std::vector<std::int64_t>{2, 2}));
    TITAN_EXPECT(f.read_f32(*bf16) == (std::vector<float>{1, -3, 2, 0}));
    std::filesystem::remove(p);
}

TITAN_TEST(Safetensors, ReadsMetadata) {
    const auto p = temp_path("titan_st_metadata.safetensors");
    // Trailing spaces pad the header, as the reference writer does.
    write_safetensors(p, {{"w", "F32", {1}, f32_bytes({4})}},
                      R"({"__metadata__":{"format":"pt"},)"
                      R"("w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})"
                      "     ");
    const SafetensorsFile f = SafetensorsFile::open(p);
    TITAN_EXPECT_EQ(f.entries().size(), std::size_t{1});
    TITAN_EXPECT_EQ(f.metadata().at("format").as_string(), std::string("pt"));
    TITAN_EXPECT(f.read_f32(f.entries()[0]) == (std::vector<float>{4}));
    std::filesystem::remove(p);
}

TITAN_TEST(Safetensors, ConvertsBf16) {
    using titan::io::bf16_to_f32;
    TITAN_EXPECT_EQ(bf16_to_f32(0x0000), 0.0F);
    TITAN_EXPECT(!std::signbit(bf16_to_f32(0x0000)));
    TITAN_EXPECT(is_negative_zero(bf16_to_f32(0x8000)));
    TITAN_EXPECT_EQ(bf16_to_f32(0x3F80), 1.0F);
    TITAN_EXPECT_EQ(bf16_to_f32(0xC040), -3.0F);
    TITAN_EXPECT_EQ(bf16_to_f32(0x0001), std::ldexp(1.0F, -133));
    TITAN_EXPECT_EQ(bf16_to_f32(0x7F80),
                    std::numeric_limits<float>::infinity());
    TITAN_EXPECT_EQ(bf16_to_f32(0xFF80),
                    -std::numeric_limits<float>::infinity());
    TITAN_EXPECT(std::isnan(bf16_to_f32(0x7FC0)));
    // A normal number with a nonzero mantissa: 0x3FC0 = 1.1b · 2^0.
    TITAN_EXPECT_EQ(bf16_to_f32(0x3FC0), 1.5F);
    // The NaN payload is kept: the 16 bits become the top half of the FP32.
    TITAN_EXPECT_EQ(std::bit_cast<std::uint32_t>(bf16_to_f32(0x7FC1)),
                    std::uint32_t{0x7FC10000});
}

TITAN_TEST(Safetensors, ConvertsF16) {
    using titan::io::f16_to_f32;
    TITAN_EXPECT_EQ(f16_to_f32(0x3C00), 1.0F);
    TITAN_EXPECT_EQ(f16_to_f32(0xC000), -2.0F);
    TITAN_EXPECT_EQ(f16_to_f32(0x0001), std::ldexp(1.0F, -24));
    TITAN_EXPECT_EQ(f16_to_f32(0x03FF), 1023.0F * std::ldexp(1.0F, -24));
    TITAN_EXPECT_EQ(f16_to_f32(0x7C00), std::numeric_limits<float>::infinity());
    TITAN_EXPECT(std::isnan(f16_to_f32(0x7E00)));
    TITAN_EXPECT(is_negative_zero(f16_to_f32(0x8000)));
    TITAN_EXPECT_EQ(f16_to_f32(0xFC00),
                    -std::numeric_limits<float>::infinity());
    // A negative subnormal keeps its sign: -1 · 2^-24.
    TITAN_EXPECT_EQ(f16_to_f32(0x8001), -std::ldexp(1.0F, -24));
    // The NaN payload 0x201 moves to the top of the 23-bit mantissa:
    // 0x201 << 13 = 0x402000.
    TITAN_EXPECT_EQ(std::bit_cast<std::uint32_t>(f16_to_f32(0x7E01)),
                    std::uint32_t{0x7FC02000});
}

TITAN_TEST(Safetensors, EmptyTensor) {
    const auto p = temp_path("titan_st_empty.safetensors");
    // The empty tensor sits between two others at [4, 4).
    write_safetensors(p, {{"a", "F32", {1}, f32_bytes({1})},
                          {"empty", "BF16", {0}, ""},
                          {"b", "F32", {1}, f32_bytes({2})}});
    const SafetensorsFile f = SafetensorsFile::open(p);
    const auto* e = f.find("empty");
    TITAN_EXPECT(e != nullptr);
    TITAN_EXPECT_EQ(e->begin, std::uint64_t{4});
    TITAN_EXPECT_EQ(e->end, std::uint64_t{4});
    TITAN_EXPECT(f.read_f32(*e).empty());
    TITAN_EXPECT(f.read_f32(*f.find("b")) == (std::vector<float>{2}));
    std::filesystem::remove(p);

    // A file holding only an empty tensor has an empty data section.
    write_safetensors(p, {{"empty", "F32", {0}, ""}});
    const SafetensorsFile only = SafetensorsFile::open(p);
    TITAN_EXPECT(only.read_f32(only.entries().at(0)).empty());
    std::filesystem::remove(p);
}

TITAN_TEST(Safetensors, RejectsShortFile) {
    TITAN_EXPECT(bytes_reject("titan_st_short.safetensors",
                              std::string(4, '\0'), {"header length"}));
}

TITAN_TEST(Safetensors, RejectsHeaderPastEnd) {
    // N = 100, but only 2 header bytes follow.
    std::string bytes(8, '\0');
    bytes[0] = 100;
    bytes += "{}";
    TITAN_EXPECT(bytes_reject("titan_st_header_past.safetensors", bytes,
                              {"header length 100", "past the end"}));
}

TITAN_TEST(Safetensors, RejectsHugeHeader) {
    // N = 2^27, over the 100 MiB cap, and nothing after the prefix: the cap
    // is checked before the length is compared with the file or allocated.
    std::string bytes(8, '\0');
    bytes[3] = 0x08;
    TITAN_EXPECT(bytes_reject(
        "titan_st_huge_header.safetensors", bytes,
        {"header length 134217728", "over the limit of 104857600 bytes"}));
}

TITAN_TEST(Safetensors, RejectsTrailingHeaderBytes) {
    TITAN_EXPECT(header_rejects("titan_st_trailing.safetensors", "{}x", 0,
                                {"header is not valid JSON"}));
}

TITAN_TEST(Safetensors, RejectsEmptyTensorName) {
    TITAN_EXPECT(header_rejects(
        "titan_st_empty_name.safetensors",
        R"({"":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", 4,
        {"empty tensor name"}));
}

TITAN_TEST(Safetensors, RejectsNonObjectHeader) {
    TITAN_EXPECT(header_rejects("titan_st_array.safetensors", "[1,2]", 0,
                                {"header", "not a JSON object"}));
}

TITAN_TEST(Safetensors, RejectsUnsupportedDtype) {
    TITAN_EXPECT(header_rejects(
        "titan_st_i8.safetensors",
        R"({"model.w":{"dtype":"I8","shape":[4],"data_offsets":[0,4]}})", 4,
        {"model.w.dtype", "I8"}));
}

TITAN_TEST(Safetensors, RejectsMissingDataOffsets) {
    TITAN_EXPECT(header_rejects("titan_st_no_offsets.safetensors",
                                R"({"model.w":{"dtype":"F32","shape":[1]}})", 4,
                                {"model.w.data_offsets", "missing"}));
}

TITAN_TEST(Safetensors, RejectsExtraEntryKey) {
    TITAN_EXPECT(header_rejects(
        "titan_st_extra_key.safetensors",
        R"({"model.w":{"dtype":"F32","shape":[1],"data_offsets":[0,4],)"
        R"("extra":1}})",
        4, {"model.w.extra", "unknown key"}));
}

TITAN_TEST(Safetensors, RejectsEndBeforeBegin) {
    TITAN_EXPECT(header_rejects(
        "titan_st_end_before.safetensors",
        R"({"model.w":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})", 4,
        {"model.w.data_offsets", "end 0 is before begin 4"}));
}

TITAN_TEST(Safetensors, RejectsEndPastData) {
    TITAN_EXPECT(header_rejects(
        "titan_st_end_past.safetensors",
        R"({"model.w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", 4,
        {"model.w.data_offsets", "past the data section"}));
}

TITAN_TEST(Safetensors, RejectsSizeShapeMismatch) {
    TITAN_EXPECT(header_rejects(
        "titan_st_size.safetensors",
        R"({"model.w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", 8,
        {"model.w.data_offsets", "8 bytes", "needs 12"}));
}

TITAN_TEST(Safetensors, RejectsOverlap) {
    TITAN_EXPECT(header_rejects(
        "titan_st_overlap.safetensors",
        R"({"model.a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
        R"("model.b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",
        12, {"data not contiguous", "model.b"}));
}

TITAN_TEST(Safetensors, RejectsGap) {
    TITAN_EXPECT(header_rejects(
        "titan_st_gap.safetensors",
        R"({"model.a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
        R"("model.b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}})",
        12, {"data not contiguous", "model.b"}));
}

TITAN_TEST(Safetensors, RejectsNegativeDimension) {
    TITAN_EXPECT(header_rejects(
        "titan_st_negative.safetensors",
        R"({"model.w":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}})", 0,
        {"model.w.shape", "-1"}));
}

TITAN_TEST(Safetensors, RejectsShapeOverflow) {
    // 2^32 · 2^32 = 2^64 elements, past the int64 range.
    TITAN_EXPECT(header_rejects(
        "titan_st_overflow.safetensors",
        R"({"model.w":{"dtype":"F32","shape":[4294967296,4294967296],)"
        R"("data_offsets":[0,0]}})",
        0, {"model.w.shape", "overflows"}));
}

TITAN_TEST(Safetensors, RejectsNonStringMetadata) {
    TITAN_EXPECT(header_rejects("titan_st_metadata_num.safetensors",
                                R"({"__metadata__":{"step":3}})", 0,
                                {"__metadata__.step", "expected a string"}));
}
