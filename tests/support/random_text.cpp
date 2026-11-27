#include "support/random_text.hpp"

#include <string_view>
#include <utility>

#include "core/rng.hpp"

namespace titan::test {

namespace {

// Picks a uniform element of `items`.
template <typename T, std::size_t N>
const T& pick(Rng& rng, const T (&items)[N]) {
    return items[static_cast<std::size_t>(
        rng.randint(static_cast<std::int64_t>(N)))];
}

constexpr std::string_view kLetters =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
constexpr std::string_view kDigits = "0123456789";
constexpr std::string_view kPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
constexpr std::string_view kWhitespace[] = {" ", "\n", "\xC2\xA0"};
constexpr std::string_view kNonAscii[] = {"\xC3\xA9", "\xE6\x97\xA5",
                                          "\xD9\xA1", "\xF0\x9F\x99\x82"};

constexpr std::int64_t kMaxSymbols = 64;

std::string_view pick_char(Rng& rng, std::string_view chars) {
    const auto i = static_cast<std::size_t>(
        rng.randint(static_cast<std::int64_t>(chars.size())));
    return chars.substr(i, 1);
}

}  // namespace

std::vector<std::string> random_texts(std::uint64_t seed, std::size_t count) {
    Rng rng(seed);
    std::vector<std::string> texts;
    texts.reserve(count);
    for (std::size_t t = 0; t < count; ++t) {
        std::string text;
        const std::int64_t symbols = rng.randint(kMaxSymbols);
        for (std::int64_t s = 0; s < symbols; ++s) {
            switch (rng.randint(5)) {
                case 0:
                    text += pick_char(rng, kLetters);
                    break;
                case 1:
                    text += pick_char(rng, kDigits);
                    break;
                case 2:
                    text += pick_char(rng, kPunctuation);
                    break;
                case 3:
                    text += pick(rng, kWhitespace);
                    break;
                default:
                    text += pick(rng, kNonAscii);
                    break;
            }
        }
        texts.push_back(std::move(text));
    }
    return texts;
}

}  // namespace titan::test
