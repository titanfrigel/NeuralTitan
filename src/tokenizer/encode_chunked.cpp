#include "tokenizer/encode_chunked.hpp"

#include <algorithm>
#include <cstddef>
#include <string_view>
#include <utility>

#include "core/parallel.hpp"

namespace titan::tokenizer {

namespace {

constexpr std::string_view kEndOfText = "<|endoftext|>";

// Splits [0, text.size()) into byte ranges that start and end right after an
// <|endoftext|> (or at the ends of the text), each of roughly text.size() /
// parts bytes: a range is extended chunk by chunk until it reaches that size,
// so a long marker-free chunk makes a longer range.
std::vector<std::pair<std::size_t, std::size_t>> split_ranges(
    std::string_view text, std::size_t parts) {
    const std::size_t target = (text.size() + parts - 1) / parts;
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t end = begin;
        while (end < text.size() && end - begin < target) {
            const std::size_t marker = text.find(kEndOfText, end);
            end = marker == std::string_view::npos ? text.size()
                                                   : marker + kEndOfText.size();
        }
        ranges.emplace_back(begin, end);
        begin = end;
    }
    return ranges;
}

}  // namespace

std::vector<std::int32_t> encode_chunked(const Tokenizer& tok,
                                         std::string_view text) {
    if (!tok.eos_id().has_value()) return tok.encode(text);

    // Why the ids equal tok.encode(text): encode matches added tokens before
    // it pre-tokenizes, so no pre-tokenized piece (and no merge) ever spans an
    // <|endoftext|>; the marker is one id, and the text on each side is
    // encoded independently. Cutting right after a marker therefore gives
    // the same ids as encoding in place. This assumes that no added token
    // starts inside "<|endoftext|>" or contains it past its first byte; the
    // SmolLM2 tokenizer and ours have none.
    const auto ranges =
        split_ranges(text, static_cast<std::size_t>(num_threads()));
    std::vector<std::vector<std::int32_t>> parts(ranges.size());
    parallel_for(0, static_cast<std::int64_t>(ranges.size()),
                 [&](std::int64_t first, std::int64_t last) {
                     for (std::int64_t i = first; i < last; ++i) {
                         const auto [begin, end] =
                             ranges[static_cast<std::size_t>(i)];
                         parts[static_cast<std::size_t>(i)] =
                             tok.encode(text.substr(begin, end - begin));
                     }
                 });

    // Concatenate in order, releasing each part as it is copied.
    std::vector<std::int32_t> ids;
    if (parts.empty()) return ids;
    std::size_t total = 0;
    for (const auto& part : parts) total += part.size();
    ids = std::move(parts[0]);
    ids.reserve(total);
    for (std::size_t i = 1; i < parts.size(); ++i) {
        ids.insert(ids.end(), parts[i].begin(), parts[i].end());
        std::vector<std::int32_t>().swap(parts[i]);
    }
    return ids;
}

}  // namespace titan::tokenizer
