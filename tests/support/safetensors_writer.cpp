#include "support/safetensors_writer.hpp"

#include <cstddef>
#include <utility>

#include "io/file.hpp"
#include "io/json.hpp"

namespace titan::test {

void write_safetensors(const std::filesystem::path& path,
                       const std::vector<TestTensor>& tensors,
                       std::string_view header_override) {
    std::string header;
    if (!header_override.empty()) {
        header = header_override;
    } else {
        io::JsonValue root = io::JsonValue::object();
        std::uint64_t offset = 0;
        for (const TestTensor& t : tensors) {
            io::JsonValue shape = io::JsonValue::array();
            for (std::int64_t d : t.shape) {
                shape.push_back(io::JsonValue::number(static_cast<double>(d)));
            }
            io::JsonValue offsets = io::JsonValue::array();
            offsets.push_back(
                io::JsonValue::number(static_cast<double>(offset)));
            offset += t.bytes.size();
            offsets.push_back(
                io::JsonValue::number(static_cast<double>(offset)));

            io::JsonValue entry = io::JsonValue::object();
            entry.set("dtype", io::JsonValue::string(t.dtype));
            entry.set("shape", std::move(shape));
            entry.set("data_offsets", std::move(offsets));
            root.set(t.name, std::move(entry));
        }
        header = io::to_json(root, 0);
        header.resize((header.size() + 7) / 8 * 8, ' ');
    }

    std::string out(8, '\0');
    for (std::size_t i = 0; i < 8; ++i) {
        out[i] = static_cast<char>((header.size() >> (8 * i)) & 0xFF);
    }
    out += header;
    for (const TestTensor& t : tensors) {
        out += t.bytes;
    }
    io::write_file(path, out);
}

}  // namespace titan::test
