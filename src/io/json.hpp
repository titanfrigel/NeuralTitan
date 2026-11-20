#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace titan::io {

// A JSON document node: null, bool, number (double), string (UTF-8), array or
// object. Objects keep their keys in insertion order and never hold a
// duplicate key.
class JsonValue {
public:
    using Array = std::vector<JsonValue>;
    using Object = std::vector<std::pair<std::string, JsonValue>>;

    // Constructs null.
    JsonValue() = default;

    static JsonValue boolean(bool value);
    static JsonValue number(double value);
    static JsonValue string(std::string value);
    static JsonValue array();
    static JsonValue object();
    // `members` must have distinct keys (the parser guarantees it); use set()
    // when building an object in code.
    static JsonValue object(Object members);

    bool is_null() const;
    bool is_bool() const;
    bool is_number() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;

    // Accessors throw titan::Error naming the expected and the actual kind.
    bool as_bool() const;
    double as_number() const;
    // Rejects non-integral numbers and magnitudes above 2^53.
    std::int64_t as_int() const;
    const std::string& as_string() const;
    const Array& as_array() const;
    const Object& as_object() const;

    // The value stored under `key`, or nullptr. Throws if not an object.
    const JsonValue* find(std::string_view key) const;
    // Like find, but throws naming the missing key.
    const JsonValue& at(std::string_view key) const;

    // Appends to an array. Throws if not an array.
    void push_back(JsonValue value);
    // Sets a key of an object, replacing an existing value. Throws if not an
    // object.
    void set(std::string key, JsonValue value);

private:
    std::variant<std::monostate, bool, double, std::string, Array, Object>
        data_;
};

// Parses `text` as one RFC 8259 JSON document. Throws titan::Error with the
// line and column of the first problem.
JsonValue parse_json(std::string_view text);

// Serialises `value`. indent > 0 pretty-prints with that many spaces per
// level; indent <= 0 writes one compact line.
std::string to_json(const JsonValue& value, int indent = 2);

}  // namespace titan::io
