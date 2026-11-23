#include "core/rng.hpp"

#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <exception>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"

namespace titan {

float Rng::uniform() {
    // 2^-24; (next_u64() >> 40) has 24 bits.
    return static_cast<float>(next_u64() >> 40) * (1.0f / 16777216.0f);
}

float Rng::normal() {
    if (has_cached_normal_) {
        has_cached_normal_ = false;
        return static_cast<float>(cached_normal_);
    }
    // u1 in (0, 1] so that log(u1) is finite; u2 in [0, 1). 53 bits each.
    constexpr double k2m53 = 1.0 / 9007199254740992.0;
    double u1 = 1.0 - static_cast<double>(next_u64() >> 11) * k2m53;
    double u2 = static_cast<double>(next_u64() >> 11) * k2m53;
    // z0 = r cos(theta), z1 = r sin(theta), r = sqrt(-2 ln u1), theta = 2 pi
    // u2.
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = kTwoPi * u2;
    cached_normal_ = r * std::sin(theta);
    has_cached_normal_ = true;
    return static_cast<float>(r * std::cos(theta));
}

std::int64_t Rng::randint(std::int64_t n) {
    TITAN_CHECK(n > 0, "randint needs n > 0, got " + std::to_string(n));
    const auto range = static_cast<std::uint64_t>(n);
    // Accept only draws below the largest multiple of `range` that fits in
    // 64 bits; each residue is then equally likely. 2^64 mod range equals
    // (UINT64_MAX % range + 1) % range.
    const std::uint64_t excess = (UINT64_MAX % range + 1) % range;
    const std::uint64_t limit = UINT64_MAX - excess;
    std::uint64_t x = next_u64();
    while (x > limit) {
        x = next_u64();
    }
    return static_cast<std::int64_t>(x % range);
}

void Rng::fill_normal(Tensor& t, float mean, float stddev) {
    TITAN_CHECK(t.device() == Device::CPU, "fill_normal needs a CPU tensor");
    float* p = t.data();
    const std::int64_t n = t.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        p[i] = mean + stddev * normal();
    }
}

std::string Rng::save_state() const {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << engine_ << ' ' << (has_cached_normal_ ? 1 : 0) << ' '
        << std::bit_cast<std::uint64_t>(cached_normal_);
    return out.str();
}

namespace {

// The whitespace-separated fields of `text`.
std::vector<std::string_view> split_fields(std::string_view text) {
    std::vector<std::string_view> fields;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() &&
               std::isspace(static_cast<unsigned char>(text[i])))
            ++i;
        const std::size_t start = i;
        while (i < text.size() &&
               !std::isspace(static_cast<unsigned char>(text[i])))
            ++i;
        if (i > start) fields.push_back(text.substr(start, i - start));
    }
    return fields;
}

// Parses a whole field as a decimal uint64.
bool parse_u64(std::string_view field, std::uint64_t& out) {
    const auto [end, ec] =
        std::from_chars(field.data(), field.data() + field.size(), out);
    return ec == std::errc() && end == field.data() + field.size();
}

}  // namespace

void Rng::load_state(std::string_view state) {
    // The standard library's engine reader does not report a bad stream
    // reliably (MSVC throws std::invalid_argument or reads garbage), so check
    // the fields here first: the engine's own field count, then digits only.
    std::ostringstream reference;
    reference << std::mt19937_64();
    const std::size_t engine_fields = split_fields(reference.str()).size();

    const std::vector<std::string_view> fields = split_fields(state);
    bool ok = fields.size() == engine_fields + 2;
    std::uint64_t value = 0;
    for (std::size_t i = 0; ok && i < fields.size(); ++i) {
        ok = parse_u64(fields[i], value);
    }
    std::uint64_t flag = 0;
    std::uint64_t bits = 0;
    ok = ok && parse_u64(fields[engine_fields], flag) && flag <= 1 &&
         parse_u64(fields[engine_fields + 1], bits);
    std::mt19937_64 engine;
    if (ok) {
        std::string engine_text;
        for (std::size_t i = 0; i < engine_fields; ++i) {
            engine_text += i == 0 ? "" : " ";
            engine_text += fields[i];
        }
        std::istringstream in(engine_text);
        in.imbue(std::locale::classic());
        try {
            in >> engine;
            ok = !in.fail();
        } catch (const std::exception&) {
            ok = false;
        }
    }
    TITAN_CHECK(ok, "Rng::load_state: malformed state");
    engine_ = engine;
    has_cached_normal_ = flag == 1;
    cached_normal_ = std::bit_cast<double>(bits);
}

}  // namespace titan
