#include "tokenizer/pretokenizer.hpp"

#include <cstddef>
#include <string>

#include "core/error.hpp"
#include "core/utf8.hpp"
#include "tokenizer/unicode_tables.hpp"

namespace titan::tokenizer {

namespace {

// The decoded text: code point i is cps[i] and starts at byte offsets[i];
// offsets has one extra entry, text.size(), so code points [a, b) span bytes
// [offsets[a], offsets[b]).
struct Decoded {
    std::vector<char32_t> cps;
    std::vector<std::size_t> offsets;
};

Decoded decode(std::string_view text) {
    Decoded d;
    std::size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = 0;
        std::size_t len = 0;
        TITAN_CHECK(
            decode_utf8_at(text, pos, cp, len),
            "pretokenize: invalid UTF-8 at byte offset " + std::to_string(pos));
        d.cps.push_back(cp);
        d.offsets.push_back(pos);
        pos += len;
    }
    d.offsets.push_back(text.size());
    return d;
}

// [^\s\p{L}\p{N}]: anything that is not whitespace, a letter or a number.
bool is_other(char32_t cp) {
    return !is_whitespace(cp) && !is_letter(cp) && !is_number(cp);
}

// Scans the GPT-2 pattern over code points [begin, end) of `cps` and returns
// the end of the match that starts at `pos` (pos < end). The alternatives are
// tried in order, as the regex engine does; every code point is whitespace, a
// letter, a number or "other", so one of them always matches.
std::size_t match_at(const std::vector<char32_t>& cps, std::size_t pos,
                     std::size_t end) {
    const auto at = [&](std::size_t i) -> char32_t {
        return i < end ? cps[i] : 0;
    };

    // 's|'t|'re|'ve|'m|'ll|'d: case-sensitive, ASCII apostrophe only.
    if (cps[pos] == U'\'' && pos + 1 < end) {
        const char32_t c1 = at(pos + 1);
        const char32_t c2 = at(pos + 2);
        if (c1 == U's' || c1 == U't' || c1 == U'm' || c1 == U'd') {
            return pos + 2;
        }
        if ((c1 == U'r' && c2 == U'e') || (c1 == U'v' && c2 == U'e') ||
            (c1 == U'l' && c2 == U'l')) {
            return pos + 3;
        }
    }

    // ` ?X+` for the classes \p{L}, \p{N} and [^\s\p{L}\p{N}]: an optional
    // U+0020 then a maximal run of the class. When the space is not followed
    // by the class, the regex backtracks to no space, which needs the class
    // at `pos` itself.
    const auto space_then_run = [&](bool (*in_class)(char32_t)) {
        std::size_t start = pos;
        if (cps[start] == U' ' && start + 1 < end && in_class(cps[start + 1])) {
            ++start;
        }
        if (!in_class(cps[start])) return pos;  // no match
        std::size_t i = start + 1;
        while (i < end && in_class(cps[i])) ++i;
        return i;
    };
    for (bool (*in_class)(char32_t) : {is_letter, is_number, is_other}) {
        const std::size_t match = space_then_run(in_class);
        if (match != pos) return match;
    }

    // Only whitespace is left at `pos`. Let [pos, run_end) be its maximal run.
    std::size_t run_end = pos + 1;
    while (run_end < end && is_whitespace(cps[run_end])) ++run_end;

    // \s+(?!\S): the longest run not followed by a non-space. At the end of
    // the piece that is the whole run. Otherwise a non-space follows, so the
    // regex backtracks one character, leaving the last whitespace to attach
    // to the next word; that needs a run of at least two.
    if (run_end == end) return run_end;
    if (run_end - pos >= 2) return run_end - 1;

    // \s+: a single whitespace before a non-space.
    return run_end;
}

}  // namespace

std::vector<std::string_view> pretokenize(std::string_view text) {
    const Decoded d = decode(text);
    const std::size_t n = d.cps.size();

    std::vector<std::string_view> pieces;
    const auto emit = [&](std::size_t a, std::size_t b) {
        pieces.push_back(
            text.substr(d.offsets[a], d.offsets[b] - d.offsets[a]));
    };
    // Stage 2 on code points [begin, end): split by the GPT-2 pattern.
    const auto split = [&](std::size_t begin, std::size_t end) {
        std::size_t pos = begin;
        while (pos < end) {
            const std::size_t next = match_at(d.cps, pos, end);
            emit(pos, next);
            pos = next;
        }
    };

    // Stage 1: every is_numeric code point is its own piece; the runs
    // between them go through stage 2. The Digits stage has its own number
    // class, a superset of \p{N} (see unicode_tables.hpp), so stage 2 never
    // sees a number; is_number stays in the scanner to keep the pattern whole.
    std::size_t run_begin = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_numeric(d.cps[i])) continue;
        split(run_begin, i);
        emit(i, i + 1);
        run_begin = i + 1;
    }
    split(run_begin, n);
    return pieces;
}

}  // namespace titan::tokenizer
