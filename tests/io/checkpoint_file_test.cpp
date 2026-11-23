#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "io/checkpoint_file.hpp"
#include "io/checksum.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "support/parity.hpp"

namespace {

using titan::Device;
using titan::Shape;
using titan::Tensor;
using titan::io::JsonValue;
using titan::io::NamedTensor;

std::filesystem::path temp_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

Tensor sequence(Shape shape, float start) {
    std::int64_t n = 1;
    for (std::int64_t d : shape) {
        n *= d;
    }
    std::vector<float> v(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = start + 0.5F * static_cast<float>(i);
    }
    return Tensor::from_vector(v, std::move(shape));
}

JsonValue sample_meta() {
    JsonValue meta = JsonValue::object();
    meta.set("step", JsonValue::number(42));
    meta.set("kind", JsonValue::string("test"));
    return meta;
}

std::vector<NamedTensor> sample_tensors() {
    return {{"t1", sequence({1}, -3.0F)},
            {"t2", sequence({3, 5}, 0.25F)},
            {"t3", sequence({2, 3, 4, 5}, 7.0F)}};
}

// The size of the last tensor of sample_tensors(), 120 floats, padded to 64.
constexpr std::size_t kLastTensorPadded = 512;

std::string read_all(const std::filesystem::path& p) {
    return titan::io::read_file(p);
}

// Runs `fn` and returns whether it threw a titan::Error whose message
// contains `needle`.
bool throws_containing(const std::function<void()>& fn,
                       const std::string& needle) {
    try {
        fn();
    } catch (const titan::Error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

bool read_rejects(const std::filesystem::path& p, const std::string& needle) {
    return throws_containing([&] { titan::io::read_checkpoint_file(p); },
                             needle);
}

void put_le(std::string& s, std::size_t pos, std::uint64_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        s[pos + static_cast<std::size_t>(i)] =
            static_cast<char>((v >> (8 * i)) & 0xFF);
    }
}

// Builds a complete container around `header` with a correct size, header
// length and header CRC, followed by `data_bytes` zero bytes of data section.
std::string raw_file(const std::string& header, std::size_t data_bytes) {
    std::string s(28, '\0');
    s.replace(0, 4, "TCKP");
    put_le(s, 4, 1, 4);
    put_le(s, 16, header.size(), 8);
    const auto* p = reinterpret_cast<const std::byte*>(header.data());
    put_le(s, 24, titan::io::crc32({p, header.size()}), 4);
    s += header;
    s.resize((s.size() + 63) / 64 * 64, '\0');
    s.resize(s.size() + data_bytes, '\0');
    put_le(s, 8, s.size(), 8);
    return s;
}

// Writes the sample checkpoint to `p` and returns its bytes.
std::string write_sample(const std::filesystem::path& p) {
    titan::io::write_checkpoint_file(p, sample_meta(), sample_tensors());
    return read_all(p);
}

bool expect_rejects_bytes(const char* name, const std::string& bytes,
                          const std::string& needle) {
    const auto p = temp_path(name);
    titan::io::write_file(p, bytes);
    const bool ok = read_rejects(p, needle);
    std::filesystem::remove(p);
    return ok;
}

}  // namespace

TITAN_TEST(Crc32, KnownVector) {
    const std::string s = "123456789";
    const auto* p = reinterpret_cast<const std::byte*>(s.data());
    TITAN_EXPECT_EQ(titan::io::crc32({p, s.size()}), 0xCBF43926U);
    // Feeding the data in two pieces gives the same result.
    const std::uint32_t head = titan::io::crc32({p, 4});
    TITAN_EXPECT_EQ(titan::io::crc32({p + 4, 5}, head), 0xCBF43926U);
}

TITAN_TEST(Checkpoint, RoundTrip) {
    const auto p = temp_path("titan_ckpt_roundtrip.tckp");
    write_sample(p);
    const auto file = titan::io::read_checkpoint_file(p);
    const auto expected = sample_tensors();

    TITAN_EXPECT_EQ(titan::io::to_json(file.meta, 0),
                    titan::io::to_json(sample_meta(), 0));
    TITAN_EXPECT_EQ(file.tensors.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        TITAN_EXPECT_EQ(file.tensors[i].name, expected[i].name);
        TITAN_EXPECT(file.tensors[i].value.shape() ==
                     expected[i].value.shape());
        TITAN_EXPECT(file.tensors[i].value.to_vector() ==
                     expected[i].value.to_vector());
    }
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, EmptyTensorListRoundTrips) {
    const auto p = temp_path("titan_ckpt_empty.tckp");
    titan::io::write_checkpoint_file(p, sample_meta(), {});
    const auto file = titan::io::read_checkpoint_file(p);
    TITAN_EXPECT(file.tensors.empty());
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, TensorsArePaddedTo64Bytes) {
    const auto p = temp_path("titan_ckpt_padding.tckp");
    const std::string bytes = write_sample(p);
    TITAN_EXPECT_EQ(bytes.size() % 64, std::size_t{0});
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, ReadMetaReturnsMeta) {
    const auto p = temp_path("titan_ckpt_meta.tckp");
    write_sample(p);
    const JsonValue meta = titan::io::read_checkpoint_meta(p);
    TITAN_EXPECT_EQ(titan::io::to_json(meta, 0),
                    titan::io::to_json(sample_meta(), 0));
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, ReadMetaRejectsHeaderChecksum) {
    const auto p = temp_path("titan_ckpt_meta_bad.tckp");
    std::string bytes = write_sample(p);
    bytes[30] = static_cast<char>(bytes[30] ^ 0x01);
    titan::io::write_file(p, bytes);
    TITAN_EXPECT(throws_containing([&] { titan::io::read_checkpoint_meta(p); },
                                   "header checksum"));
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, AtomicReplaceKeepsOldOnFailure) {
    const auto p = temp_path("titan_ckpt_atomic.tckp");
    const auto tmp = temp_path("titan_ckpt_atomic.tckp.tmp");
    const std::string a = write_sample(p);
    TITAN_EXPECT(!std::filesystem::exists(tmp));

    // A failure while writing B: the temporary file cannot be created because
    // its name is taken by a directory.
    std::filesystem::create_directory(tmp);
    TITAN_EXPECT_THROWS(titan::io::write_checkpoint_file(
                            p, sample_meta(),
                            std::vector<NamedTensor>{{"x", sequence({2}, 1)}}),
                        titan::Error);
    std::filesystem::remove(tmp);

    TITAN_EXPECT(read_all(p) == a);
    const auto file = titan::io::read_checkpoint_file(p);
    TITAN_EXPECT_EQ(file.tensors.size(), std::size_t{3});

    // A failure while validating B: nothing is created at all.
    const std::vector<NamedTensor> dup = {{"x", sequence({2}, 1)},
                                          {"x", sequence({2}, 2)}};
    TITAN_EXPECT_THROWS(titan::io::write_checkpoint_file(p, sample_meta(), dup),
                        titan::Error);
    TITAN_EXPECT(!std::filesystem::exists(tmp));
    TITAN_EXPECT(read_all(p) == a);

    // A successful write replaces A and leaves no temporary file.
    titan::io::write_checkpoint_file(
        p, sample_meta(), std::vector<NamedTensor>{{"x", sequence({2}, 1)}});
    TITAN_EXPECT(!std::filesystem::exists(tmp));
    TITAN_EXPECT_EQ(titan::io::read_checkpoint_file(p).tensors.size(),
                    std::size_t{1});
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, WriterRejectsCudaTensorNamingIt) {
    TITAN_REQUIRE_CUDA();
    const auto p = temp_path("titan_ckpt_cuda.tckp");
    const std::string a = write_sample(p);
    const std::vector<NamedTensor> bad = {
        {"on_gpu", Tensor::zeros({2}, Device::CUDA)}};
    TITAN_EXPECT(throws_containing(
        [&] { titan::io::write_checkpoint_file(p, sample_meta(), bad); },
        "'on_gpu' is not on the CPU"));
    TITAN_EXPECT(read_all(p) == a);
    TITAN_EXPECT(
        !std::filesystem::exists(temp_path("titan_ckpt_cuda.tckp.tmp")));
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, RejectsTruncated) {
    const auto p = temp_path("titan_ckpt_trunc.tckp");
    const std::string bytes = write_sample(p);
    for (std::size_t i = 0; i < 8; ++i) {
        const std::size_t cut = i * (bytes.size() - 1) / 7;
        TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_trunc_cut.tckp",
                                          bytes.substr(0, cut), "size"));
    }
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, RejectsWrongMagic) {
    const auto p = temp_path("titan_ckpt_magic.tckp");
    std::string bytes = write_sample(p);
    bytes[0] = 'X';
    TITAN_EXPECT(
        expect_rejects_bytes("titan_ckpt_magic_bad.tckp", bytes, "bad magic"));
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, RejectsWrongVersion) {
    const auto p = temp_path("titan_ckpt_version.tckp");
    std::string bytes = write_sample(p);
    put_le(bytes, 4, 2, 4);
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_version_bad.tckp", bytes,
                                      "unsupported version 2"));
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, RejectsSizeMismatch) {
    const auto p = temp_path("titan_ckpt_size.tckp");
    std::string bytes = write_sample(p);
    put_le(bytes, 8, bytes.size() + 1, 8);
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_size_bad.tckp", bytes,
                                      "total size mismatch"));
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, RejectsHeaderTooLarge) {
    const auto p = temp_path("titan_ckpt_hlen.tckp");
    std::string bytes = write_sample(p);
    put_le(bytes, 16, (std::uint64_t{16} << 20) + 1, 8);
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_hlen_big.tckp", bytes,
                                      "header length"));
    put_le(bytes, 16, 0, 8);
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_hlen_zero.tckp", bytes,
                                      "header length"));
    // Within the cap but past the end of the file.
    put_le(bytes, 16, bytes.size(), 8);
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_hlen_past.tckp", bytes,
                                      "past the end"));
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, RejectsHeaderChecksum) {
    const auto p = temp_path("titan_ckpt_hcrc.tckp");
    std::string bytes = write_sample(p);
    bytes[30] = static_cast<char>(bytes[30] ^ 0x01);
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_hcrc_bad.tckp", bytes,
                                      "header checksum"));
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, RejectsTensorChecksum) {
    const auto p = temp_path("titan_ckpt_tcrc.tckp");
    std::string bytes = write_sample(p);
    // The first byte of the last tensor, "t3".
    const std::size_t pos = bytes.size() - kLastTensorPadded;
    bytes[pos] = static_cast<char>(bytes[pos] ^ 0x01);
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_tcrc_bad.tckp", bytes,
                                      "tensor 't3': checksum mismatch"));
    std::filesystem::remove(p);
}

TITAN_TEST(Checkpoint, RejectsTensorOutOfBounds) {
    const std::string header =
        R"({"meta":{},"tensors":[{"name":"a","shape":[2],"offset":64,"crc32":0}]})";
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_oob.tckp",
                                      raw_file(header, 64),
                                      "tensor 'a': out of bounds"));
    // A huge offset must not overflow the bounds check.
    const std::string huge =
        R"({"meta":{},"tensors":[{"name":"a","shape":[2],"offset":9007199254740991,"crc32":0}]})";
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_oob_huge.tckp",
                                      raw_file(huge, 64),
                                      "tensor 'a': out of bounds"));
    // A shape larger than the data section.
    const std::string big =
        R"({"meta":{},"tensors":[{"name":"a","shape":[100],"offset":0,"crc32":0}]})";
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_oob_big.tckp",
                                      raw_file(big, 64),
                                      "tensor 'a': out of bounds"));
}

TITAN_TEST(Checkpoint, RejectsDuplicateName) {
    const std::string header =
        R"({"meta":{},"tensors":[{"name":"a","shape":[2],"offset":0,"crc32":0},)"
        R"({"name":"a","shape":[2],"offset":64,"crc32":0}]})";
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_dup.tckp",
                                      raw_file(header, 128),
                                      "duplicate tensor name 'a'"));
}

TITAN_TEST(Checkpoint, RejectsBadShape) {
    const auto with_shape = [](const std::string& shape) {
        return std::string(R"({"meta":{},"tensors":[{"name":"a","shape":)") +
               shape + R"(,"offset":0,"crc32":0}]})";
    };
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_shape0.tckp",
                                      raw_file(with_shape("[0]"), 64),
                                      "tensor 'a': bad shape"));
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_shapeneg.tckp",
                                      raw_file(with_shape("[2,-3]"), 64),
                                      "tensor 'a': bad shape"));
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_rank5.tckp",
                                      raw_file(with_shape("[1,1,1,1,1]"), 64),
                                      "tensor 'a': bad shape"));
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_rank0.tckp",
                                      raw_file(with_shape("[]"), 64),
                                      "tensor 'a': bad shape"));
    TITAN_EXPECT(expect_rejects_bytes(
        "titan_ckpt_overflow.tckp",
        raw_file(with_shape("[4503599627370496,4503599627370496]"), 64),
        "byte size overflows"));
}

TITAN_TEST(Checkpoint, RejectsMalformedHeaderJson) {
    TITAN_EXPECT(expect_rejects_bytes(
        "titan_ckpt_badjson.tckp", raw_file("{not json", 0), "not valid JSON"));
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_nometa.tckp",
                                      raw_file(R"({"tensors":[]})", 0),
                                      "no 'meta' object"));
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_notensors.tckp",
                                      raw_file(R"({"meta":{}})", 0),
                                      "no 'tensors' array"));
}

TITAN_TEST(Checkpoint, RejectsTrailingBytes) {
    const std::string header =
        R"({"meta":{},"tensors":[{"name":"a","shape":[2],"offset":0,"crc32":0}]})";
    TITAN_EXPECT(expect_rejects_bytes("titan_ckpt_trailing.tckp",
                                      raw_file(header, 128), "tensors fill"));
}

TITAN_TEST(Checkpoint, RejectsMissingFile) {
    TITAN_EXPECT(read_rejects(temp_path("titan_ckpt_does_not_exist.tckp"),
                              "cannot stat"));
}
