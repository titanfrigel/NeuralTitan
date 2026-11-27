#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace titan::test {

// One tensor for write_safetensors. `dtype` is written as given, so a test can
// use one the reader rejects; `bytes` is the raw data, which is not checked
// against `dtype` or `shape`.
struct TestTensor {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string bytes;
};

// Writes a safetensors file: the u64 little-endian header length, the JSON
// header and the tensors' bytes concatenated in the given order.
//
// Without an override, the header lists every tensor with its dtype, shape and
// packed data_offsets, and is padded with spaces to a multiple of 8 bytes, as
// the reference library writes it. A non-empty `header_override` is written
// verbatim instead, so the malformed-input tests can hand-write any header;
// the data section is still the tensors' bytes.
void write_safetensors(const std::filesystem::path& path,
                       const std::vector<TestTensor>& tensors,
                       std::string_view header_override = {});

}  // namespace titan::test
