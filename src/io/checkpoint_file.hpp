#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "core/tensor.hpp"
#include "io/json.hpp"

namespace titan::io {

// A tensor stored under a unique name. Values live on the CPU.
struct NamedTensor {
    std::string name;
    Tensor value;
};

// Checkpoint container layout, all integers little-endian:
//   bytes 0..3    ASCII magic "TCKP"
//   bytes 4..7    u32 version, currently 1
//   bytes 8..15   u64 total file size
//   bytes 16..23  u64 header length H, 1 <= H <= 16 MiB
//   bytes 24..27  u32 CRC-32 of the header bytes
//   bytes 28..    H bytes of UTF-8 JSON:
//                   {"meta": {...},
//                    "tensors": [{"name", "shape", "offset", "crc32"}, ...]}
//   then          zero padding up to a multiple of 64; the data section
//                 starts there
//   data          each tensor's float32 values at its offset (relative to the
//                 data section, a multiple of 64), in the order listed, each
//                 padded with zeros to a multiple of 64 bytes
// The file ends exactly at the end of the last tensor's padding.
//
// Writes to "<path>.tmp" and renames it over `path` once complete, so an
// existing file at `path` survives any failure. Throws titan::Error for a
// tensor off the CPU, a duplicate or empty name, or an I/O failure. `meta`
// must be a JSON object.
void write_checkpoint_file(const std::filesystem::path& path,
                           const JsonValue& meta,
                           std::span<const NamedTensor> tensors);

struct CheckpointFile {
    JsonValue meta;
    std::vector<NamedTensor> tensors;
};

// Reads and fully verifies a checkpoint: magic, version, total size, header
// length and CRC, every entry's bounds and shape, and every tensor's CRC.
// Treats the file as untrusted: throws titan::Error naming the file (and the
// tensor where one is at fault), never allocates beyond the file's size.
CheckpointFile read_checkpoint_file(const std::filesystem::path& path);

// Runs the same file-level and header checks as read_checkpoint_file but
// returns only the `meta` object, without reading any tensor data.
JsonValue read_checkpoint_meta(const std::filesystem::path& path);

// A tensor entry of the header: its name and shape, checked against the file
// size like every entry.
struct TensorInfo {
    std::string name;
    Shape shape;
};

struct CheckpointHeader {
    JsonValue meta;
    std::vector<TensorInfo> tensors;
};

// Like read_checkpoint_meta, but also returns every tensor entry, so a loader
// can compare the sizes a config implies with what the file holds before it
// allocates anything. The entries' element counts sum to at most the file
// size / 4.
CheckpointHeader read_checkpoint_header(const std::filesystem::path& path);

}  // namespace titan::io
