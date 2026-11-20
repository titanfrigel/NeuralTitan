#include "model/config.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "core/error.hpp"

namespace titan::model {

namespace {

// Largest integer a double holds exactly; JSON numbers are doubles.
constexpr double kMaxExactInt = 9007199254740992.0;  // 2^53

[[noreturn]] void config_error(const std::string& path,
                               const std::string& message) {
    throw_error(__FILE__, __LINE__, "model config: " + path + ": " + message);
}

const char* kind_name(const io::JsonValue& v) {
    if (v.is_null()) return "null";
    if (v.is_bool()) return "a bool";
    if (v.is_number()) return "a number";
    if (v.is_string()) return "a string";
    if (v.is_array()) return "an array";
    return "an object";
}

// The members of one JSON object, with every key checked against the allowed
// set so that a misspelt key fails instead of being ignored.
class Fields {
public:
    Fields(const io::JsonValue& v, std::string path,
           std::initializer_list<std::string_view> allowed)
        : value_(v), path_(std::move(path)) {
        if (!v.is_object()) {
            config_error(
                path_, std::string("expected an object, got ") + kind_name(v));
        }
        for (const auto& [key, member] : v.as_object()) {
            bool known = false;
            for (std::string_view a : allowed) {
                known = known || key == a;
            }
            if (!known) {
                config_error(path_of(key), "unknown key");
            }
        }
    }

    std::string path_of(std::string_view key) const {
        return path_ + "." + std::string(key);
    }
    const io::JsonValue* find(std::string_view key) const {
        return value_.find(key);
    }
    const io::JsonValue& require(std::string_view key) const {
        const io::JsonValue* v = value_.find(key);
        if (v == nullptr) {
            config_error(path_of(key), "missing required field");
        }
        return *v;
    }

private:
    const io::JsonValue& value_;
    std::string path_;
};

void require_positive(std::int64_t value, const std::string& path) {
    if (value < 1) {
        config_error(path, "must be >= 1, got " + std::to_string(value));
    }
}

// An integral JSON number >= 1.
std::int64_t read_size(const io::JsonValue& v, const std::string& path) {
    if (!v.is_number()) {
        config_error(path,
                     std::string("expected an integer, got ") + kind_name(v));
    }
    const double d = v.as_number();
    if (d != std::trunc(d) || std::fabs(d) > kMaxExactInt) {
        config_error(path, "expected an integer, got " + std::to_string(d));
    }
    const auto n = static_cast<std::int64_t>(d);
    require_positive(n, path);
    return n;
}

// A finite JSON number > 0 that fits in a float and stays > 0 as one.
float read_positive_float(const io::JsonValue& v, const std::string& path) {
    if (!v.is_number()) {
        config_error(path,
                     std::string("expected a number, got ") + kind_name(v));
    }
    const double d = v.as_number();
    if (!(d > 0.0) || d > std::numeric_limits<float>::max()) {
        config_error(path,
                     "must be a positive float, got " + std::to_string(d));
    }
    const float f = static_cast<float>(d);
    // A tiny positive double such as 1e-50 rounds to 0.0f.
    if (f == 0.0f) {
        config_error(path, "is too small for a float and rounds to zero");
    }
    return f;
}

bool read_bool(const io::JsonValue& v, const std::string& path) {
    if (!v.is_bool()) {
        config_error(path, std::string("expected a bool, got ") + kind_name(v));
    }
    return v.as_bool();
}

// A string that must be one of `choices`.
std::string read_choice(const io::JsonValue& v, const std::string& path,
                        std::initializer_list<std::string_view> choices) {
    if (!v.is_string()) {
        config_error(path,
                     std::string("expected a string, got ") + kind_name(v));
    }
    const std::string& s = v.as_string();
    std::string allowed;
    for (std::string_view c : choices) {
        if (s == c) return s;
        allowed += (allowed.empty() ? "" : ", ") + std::string(c);
    }
    config_error(path,
                 "unknown value \"" + s + "\", expected one of: " + allowed);
}

// The head checks Attention also makes, reported with the config path:
// dim % n_heads == 0, n_heads % n_kv_heads == 0, and an even head dimension
// with RoPE (RoPE rotates pairs (i, i + hd/2)).
void validate_heads(std::int64_t dim, const BlockConfig& b,
                    const std::string& path) {
    if (dim % b.n_heads != 0) {
        config_error(path + ".n_heads", "dim " + std::to_string(dim) +
                                            " is not divisible by n_heads " +
                                            std::to_string(b.n_heads));
    }
    if (b.n_heads % b.n_kv_heads != 0) {
        config_error(path + ".n_kv_heads",
                     "n_heads " + std::to_string(b.n_heads) +
                         " is not divisible by n_kv_heads " +
                         std::to_string(b.n_kv_heads));
    }
    const std::int64_t head_dim = dim / b.n_heads;
    if (b.position == "rope" && head_dim % 2 != 0) {
        config_error(path + ".position",
                     "rope needs an even head dimension, dim / n_heads = " +
                         std::to_string(head_dim));
    }
}

BlockConfig parse_block(const io::JsonValue& v, std::int64_t dim,
                        const std::string& path) {
    const Fields f(
        v, path,
        {"n_heads", "n_kv_heads", "mlp", "mlp_hidden", "norm", "position"});
    BlockConfig b;
    b.n_heads = read_size(f.require("n_heads"), f.path_of("n_heads"));
    b.n_kv_heads = read_size(f.require("n_kv_heads"), f.path_of("n_kv_heads"));
    b.mlp_hidden = read_size(f.require("mlp_hidden"), f.path_of("mlp_hidden"));
    b.mlp = read_choice(f.require("mlp"), f.path_of("mlp"), {"swiglu", "gelu"});
    b.norm = read_choice(f.require("norm"), f.path_of("norm"),
                         {"rmsnorm", "layernorm"});
    b.position = read_choice(f.require("position"), f.path_of("position"),
                             {"rope", "none"});
    validate_heads(dim, b, path);
    return b;
}

}  // namespace

ModelConfig parse_model_config(const io::JsonValue& model_section,
                               std::int64_t vocab_size) {
    TITAN_CHECK(vocab_size >= 1,
                "parse_model_config: vocab_size must be >= 1, got " +
                    std::to_string(vocab_size));
    const Fields f(model_section, "model",
                   {"dim", "context_length", "n_layers", "block", "blocks",
                    "rope_theta", "norm_eps", "tied_embeddings", "final_norm"});

    ModelConfig c;
    c.vocab_size = vocab_size;
    c.dim = read_size(f.require("dim"), f.path_of("dim"));
    c.context_length =
        read_size(f.require("context_length"), f.path_of("context_length"));
    if (const io::JsonValue* v = f.find("rope_theta")) {
        c.rope_theta = read_positive_float(*v, f.path_of("rope_theta"));
    }
    if (const io::JsonValue* v = f.find("norm_eps")) {
        c.norm_eps = read_positive_float(*v, f.path_of("norm_eps"));
    }
    if (const io::JsonValue* v = f.find("tied_embeddings")) {
        c.tied_embeddings = read_bool(*v, f.path_of("tied_embeddings"));
    }
    if (const io::JsonValue* v = f.find("final_norm")) {
        c.final_norm =
            read_choice(*v, f.path_of("final_norm"), {"rmsnorm", "layernorm"});
    }

    // Either "blocks": [...] or "n_layers" with one "block", never both.
    const io::JsonValue* blocks = f.find("blocks");
    const io::JsonValue* n_layers = f.find("n_layers");
    const io::JsonValue* block = f.find("block");
    if (blocks != nullptr) {
        if (n_layers != nullptr || block != nullptr) {
            config_error(f.path_of(n_layers != nullptr ? "n_layers" : "block"),
                         "cannot be combined with model.blocks");
        }
        if (!blocks->is_array()) {
            config_error(
                f.path_of("blocks"),
                std::string("expected an array, got ") + kind_name(*blocks));
        }
        const io::JsonValue::Array& items = blocks->as_array();
        if (items.empty()) {
            config_error(f.path_of("blocks"), "must not be empty");
        }
        for (std::size_t i = 0; i < items.size(); ++i) {
            c.blocks.push_back(parse_block(
                items[i], c.dim,
                f.path_of("blocks") + "[" + std::to_string(i) + "]"));
        }
    } else if (n_layers != nullptr || block != nullptr) {
        const std::int64_t n =
            read_size(f.require("n_layers"), f.path_of("n_layers"));
        const BlockConfig b =
            parse_block(f.require("block"), c.dim, f.path_of("block"));
        c.blocks.assign(static_cast<std::size_t>(n), b);
    } else {
        config_error(f.path_of("blocks"),
                     "missing; give either \"blocks\" or \"n_layers\" with "
                     "\"block\"");
    }
    return c;
}

ModelConfig llama_style(std::int64_t vocab, std::int64_t dim,
                        std::int64_t n_layers, std::int64_t n_heads,
                        std::int64_t n_kv_heads, std::int64_t context) {
    require_positive(vocab, "llama_style.vocab");
    require_positive(dim, "llama_style.dim");
    require_positive(n_layers, "llama_style.n_layers");
    require_positive(n_heads, "llama_style.n_heads");
    require_positive(n_kv_heads, "llama_style.n_kv_heads");
    require_positive(context, "llama_style.context");

    // SwiGLU has three projections instead of two, so Llama shrinks the
    // hidden size from 4·dim to 2/3 of it, 8·dim/3, keeping the parameter
    // count of a 4·dim GELU MLP, then rounds up to a multiple of 8.
    const std::int64_t hidden = (8 * dim / 3 + 7) / 8 * 8;
    BlockConfig b{n_heads, n_kv_heads, hidden, "swiglu", "rmsnorm", "rope"};
    validate_heads(dim, b, "llama_style");

    ModelConfig c;
    c.vocab_size = vocab;
    c.dim = dim;
    c.context_length = context;
    c.blocks.assign(static_cast<std::size_t>(n_layers), b);
    return c;
}

}  // namespace titan::model
