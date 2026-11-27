#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "io/json.hpp"

namespace titan::io {

// The largest JSON header open() accepts. A header this size already describes
// hundreds of thousands of tensors; the cap stops a corrupt length from
// allocating gigabytes before anything is parsed (threat T1). The reference
// `safetensors` library caps it at 100,000,000 bytes, slightly less.
constexpr std::uint64_t kMaxSafetensorsHeaderBytes = std::uint64_t{100} << 20;

// The element types the reader converts to FP32.
enum class SafetensorsDtype { F32, F16, BF16 };

// One tensor of the header. Offsets are byte offsets relative to the start of
// the data section, which follows the header; `end - begin` is the element
// count times the dtype's size.
struct SafetensorsEntry {
    std::string name;
    SafetensorsDtype dtype = SafetensorsDtype::F32;
    std::vector<std::int64_t> shape;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

// A Hugging Face safetensors file, all integers little-endian:
//   bytes 0..7    u64 header length N
//   bytes 8..     N bytes of UTF-8 JSON, padded with trailing spaces:
//                   {"<name>": {"dtype": "BF16", "shape": [d0, ...],
//                               "data_offsets": [begin, end]},
//                    ..., "__metadata__": {"<key>": "<string>", ...}}
//   then          the data section: every tensor's raw row-major values,
//                 packed with no gaps or overlaps
//
// open() reads only the header and treats the file as untrusted: it checks the
// header length against the file size and the cap, then every entry (exactly
// the keys dtype, shape and data_offsets; a supported dtype; non-negative
// dimensions whose product does not overflow; a byte range that matches the
// shape and lies inside the data section) and finally that the ranges, sorted
// by begin, tile the data section exactly. Each failure throws titan::Error
// naming the file and, where one applies, the tensor.
//
// read_f32 opens the file again for every call and reads only that tensor's
// byte range. Keeping no stream open makes the object cheap to move and lets
// a const object be read from several places; one open per tensor costs
// nothing next to the read itself.
class SafetensorsFile {
public:
    static SafetensorsFile open(const std::filesystem::path& path);

    const std::filesystem::path& path() const { return path_; }
    // Sorted by name.
    const std::vector<SafetensorsEntry>& entries() const { return entries_; }
    // The entry called `name`, or nullptr.
    const SafetensorsEntry* find(std::string_view name) const;
    // Reads `entry`, which must come from this file's entries(), and converts
    // its values to FP32. Throws titan::Error naming the file and the tensor if
    // the file can no longer be read (for example, it shrank since open()).
    std::vector<float> read_f32(const SafetensorsEntry& entry) const;
    // The "__metadata__" object, whose values are all strings, or an empty
    // object when the header has none.
    const JsonValue& metadata() const { return metadata_; }

private:
    SafetensorsFile() = default;

    std::filesystem::path path_;
    std::uint64_t data_start_ = 0;  // absolute offset of the data section
    std::vector<SafetensorsEntry> entries_;
    JsonValue metadata_;
};

// Exact conversions of a 16-bit float, given as its bit pattern, to FP32.
// See docs/math/hf-llama.md.
float bf16_to_f32(std::uint16_t bits);
float f16_to_f32(std::uint16_t bits);

}  // namespace titan::io
