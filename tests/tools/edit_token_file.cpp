// Test tool for the CLI resume tests: copies a token file and replaces one
// token, so a test can build a well-formed token file whose content is wrong.
//
//   edit_token_file <in.bin> <out.bin> <index> <id>
//
// The id is written as given, even outside any vocabulary.
#include <charconv>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

#include "io/token_file.hpp"

namespace {

template <typename T>
bool parse(std::string_view text, T& value) {
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    return ec == std::errc{} && ptr == end;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t index = 0;
    std::int32_t id = 0;
    if (argc != 5 || !parse(argv[3], index) || !parse(argv[4], id)) {
        std::cerr << "usage: edit_token_file <in.bin> <out.bin> <index> <id>\n";
        return 2;
    }
    try {
        std::vector<std::int32_t> ids = titan::io::read_token_file(argv[1]);
        if (index >= ids.size()) {
            std::cerr << "index " << index << " is past the " << ids.size()
                      << " tokens\n";
            return 1;
        }
        ids[index] = id;
        titan::io::write_token_file(argv[2], ids);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
