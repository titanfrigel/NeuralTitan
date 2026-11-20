#include "io/json_fields.hpp"

#include <cmath>
#include <limits>
#include <utility>

#include "core/error.hpp"

namespace titan::io {

namespace {

// Largest integer a double holds exactly; JSON numbers are doubles.
constexpr double kMaxExactInt = 9007199254740992.0;  // 2^53

}  // namespace

const char* json_kind_name(const JsonValue& v) {
    if (v.is_null()) return "null";
    if (v.is_bool()) return "a bool";
    if (v.is_number()) return "a number";
    if (v.is_string()) return "a string";
    if (v.is_array()) return "an array";
    return "an object";
}

JsonFields::JsonFields(const JsonValue& value, std::string context,
                       std::string path,
                       std::initializer_list<std::string_view> allowed)
    : value_(value), context_(std::move(context)), path_(std::move(path)) {
    if (!value.is_object()) {
        fail(path_,
             std::string("expected an object, got ") + json_kind_name(value));
    }
    for (const auto& [key, member] : value.as_object()) {
        bool known = false;
        for (std::string_view a : allowed) {
            known = known || key == a;
        }
        if (!known) {
            fail(path_of(key), "unknown key");
        }
    }
}

std::string JsonFields::path_of(std::string_view key) const {
    return path_ + "." + std::string(key);
}

const JsonValue* JsonFields::find(std::string_view key) const {
    return value_.find(key);
}

const JsonValue& JsonFields::require(std::string_view key) const {
    const JsonValue* v = value_.find(key);
    if (v == nullptr) {
        fail(path_of(key), "missing required field");
    }
    return *v;
}

void JsonFields::fail(const std::string& path,
                      const std::string& message) const {
    throw_error(__FILE__, __LINE__, context_ + ": " + path + ": " + message);
}

std::int64_t JsonFields::integer(const JsonValue& v, std::string_view key,
                                 std::int64_t min_value) const {
    const std::string path = path_of(key);
    if (!v.is_number()) {
        fail(path,
             std::string("expected an integer, got ") + json_kind_name(v));
    }
    const double d = v.as_number();
    if (d != std::trunc(d) || std::fabs(d) > kMaxExactInt) {
        fail(path, "expected an integer, got " + std::to_string(d));
    }
    const auto n = static_cast<std::int64_t>(d);
    if (n < min_value) {
        fail(path, "must be >= " + std::to_string(min_value) + ", got " +
                       std::to_string(n));
    }
    return n;
}

double JsonFields::number(const JsonValue& v, std::string_view key) const {
    const std::string path = path_of(key);
    if (!v.is_number()) {
        fail(path, std::string("expected a number, got ") + json_kind_name(v));
    }
    const double d = v.as_number();
    if (!std::isfinite(d) || std::fabs(d) > std::numeric_limits<float>::max()) {
        fail(path, "must be a finite float, got " + std::to_string(d));
    }
    return d;
}

float JsonFields::positive_float(const JsonValue& v,
                                 std::string_view key) const {
    const double d = number(v, key);
    if (!(d > 0.0)) {
        fail(path_of(key),
             "must be a positive float, got " + std::to_string(d));
    }
    const auto f = static_cast<float>(d);
    // A tiny positive double such as 1e-50 rounds to 0.0f.
    if (f == 0.0f) {
        fail(path_of(key), "is too small for a float and rounds to zero");
    }
    return f;
}

bool JsonFields::boolean(const JsonValue& v, std::string_view key) const {
    if (!v.is_bool()) {
        fail(path_of(key),
             std::string("expected a bool, got ") + json_kind_name(v));
    }
    return v.as_bool();
}

std::string JsonFields::choice(
    const JsonValue& v, std::string_view key,
    std::initializer_list<std::string_view> choices) const {
    if (!v.is_string()) {
        fail(path_of(key),
             std::string("expected a string, got ") + json_kind_name(v));
    }
    const std::string& s = v.as_string();
    std::string allowed;
    for (std::string_view c : choices) {
        if (s == c) return s;
        allowed += (allowed.empty() ? "" : ", ") + std::string(c);
    }
    fail(path_of(key),
         "unknown value \"" + s + "\", expected one of: " + allowed);
}

}  // namespace titan::io
