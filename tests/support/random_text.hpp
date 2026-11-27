#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace titan::test {

// `count` random UTF-8 strings of 0 to 63 symbols, drawn by an Rng seeded with
// `seed`. Each symbol first picks one of five groups uniformly, then a member
// of the group uniformly:
//   - ASCII letters;
//   - ASCII digits;
//   - ASCII punctuation;
//   - whitespace: ' ', '\n' and U+00A0;
//   - non-ASCII: U+00E9, U+65E5, U+0661 and U+1F642 (e acute, a CJK
//     ideograph, an Arabic-Indic digit and an emoji).
// Picking the group first keeps whitespace runs and multi-byte characters
// common, so tokenizer tests see their edge cases.
std::vector<std::string> random_texts(std::uint64_t seed, std::size_t count);

}  // namespace titan::test
