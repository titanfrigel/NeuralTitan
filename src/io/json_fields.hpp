#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

#include "io/json.hpp"

namespace titan::io {

// "null", "a bool", "a number", "a string", "an array" or "an object", for
// error messages.
const char* json_kind_name(const JsonValue& v);

// A strict reader for one JSON object of a config file. The constructor
// rejects a non-object and any key outside `allowed`, so a misspelt key fails
// instead of being ignored. Every reader throws titan::Error of the form
// "<context>: <path>.<key>: <problem>", e.g. "train config: train.lr: ...".
class JsonFields {
public:
    // `context` names the kind of config ("model config"), `path` the object's
    // place in the file ("model", "model.blocks[2]"), or is empty for the
    // file's top-level object, whose keys are then their own paths.
    JsonFields(const JsonValue& value, std::string context, std::string path,
               std::initializer_list<std::string_view> allowed);

    std::string path_of(std::string_view key) const;
    // The member `key`, or nullptr.
    const JsonValue* find(std::string_view key) const;
    // The member `key`; throws "missing required field" naming its path.
    const JsonValue& require(std::string_view key) const;

    // Throws titan::Error with this reader's context and the given full path.
    [[noreturn]] void fail(const std::string& path,
                           const std::string& message) const;

    // An integral number >= min_value (magnitude at most 2^53).
    std::int64_t integer(const JsonValue& v, std::string_view key,
                         std::int64_t min_value) const;
    // A finite number whose magnitude fits in a float.
    double number(const JsonValue& v, std::string_view key) const;
    // A float > 0 that does not round to zero.
    float positive_float(const JsonValue& v, std::string_view key) const;
    bool boolean(const JsonValue& v, std::string_view key) const;
    // A string equal to one of `choices`.
    std::string choice(const JsonValue& v, std::string_view key,
                       std::initializer_list<std::string_view> choices) const;

private:
    const JsonValue& value_;
    std::string context_;
    std::string path_;
};

}  // namespace titan::io
