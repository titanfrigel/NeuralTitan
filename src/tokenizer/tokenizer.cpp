#include "tokenizer/tokenizer.hpp"

#include <memory>
#include <string>

#include "core/error.hpp"
#include "tokenizer/bpe_tokenizer.hpp"
#include "tokenizer/char_tokenizer.hpp"

namespace titan::tokenizer {

namespace {

// A short description of a JSON value's kind for error messages.
std::string kind_of(const io::JsonValue& v) {
    if (v.is_null()) return "null";
    if (v.is_bool()) return "a boolean";
    if (v.is_number()) return "a number";
    if (v.is_string()) return "a string";
    if (v.is_array()) return "an array";
    return "an object";
}

// What an unsupported document holds instead of a known tokenizer layout.
std::string describe(const io::JsonValue& json) {
    if (!json.is_object()) return kind_of(json);
    if (const io::JsonValue* type = json.find("type")) {
        return type->is_string() ? "type \"" + type->as_string() + "\""
                                 : "type is " + kind_of(*type);
    }
    if (const io::JsonValue* model = json.find("model")) {
        return "model is " + kind_of(*model);
    }
    return "no \"type\" or \"model\" key";
}

}  // namespace

std::unique_ptr<Tokenizer> load_tokenizer(const io::JsonValue& json) {
    if (json.is_object()) {
        const io::JsonValue* type = json.find("type");
        if (type != nullptr && type->is_string() &&
            type->as_string() == "char") {
            return std::make_unique<CharTokenizer>(
                CharTokenizer::from_json(json));
        }
        const io::JsonValue* model = json.find("model");
        if (model != nullptr && model->is_object()) {
            return std::make_unique<BpeTokenizer>(
                BpeTokenizer::from_json(json));
        }
    }
    throw_error(__FILE__, __LINE__,
                "unsupported tokenizer document: " + describe(json));
}

}  // namespace titan::tokenizer
