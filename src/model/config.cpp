#include "model/config.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "core/error.hpp"
#include "io/json_fields.hpp"

namespace titan::model {

namespace {

[[noreturn]] void config_error(const std::string& path,
                               const std::string& message) {
    throw_error(__FILE__, __LINE__, "model config: " + path + ": " + message);
}

void require_positive(std::int64_t value, const std::string& path) {
    if (value < 1) {
        config_error(path, "must be >= 1, got " + std::to_string(value));
    }
}

io::JsonFields fields(const io::JsonValue& v, const std::string& path,
                      std::initializer_list<std::string_view> allowed) {
    return io::JsonFields(v, "model config", path, allowed);
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
    const io::JsonFields f = fields(
        v, path,
        {"n_heads", "n_kv_heads", "mlp", "mlp_hidden", "norm", "position"});
    BlockConfig b;
    b.n_heads = f.integer(f.require("n_heads"), "n_heads", 1);
    b.n_kv_heads = f.integer(f.require("n_kv_heads"), "n_kv_heads", 1);
    b.mlp_hidden = f.integer(f.require("mlp_hidden"), "mlp_hidden", 1);
    b.mlp = f.choice(f.require("mlp"), "mlp", {"swiglu", "gelu"});
    b.norm = f.choice(f.require("norm"), "norm", {"rmsnorm", "layernorm"});
    b.position = f.choice(f.require("position"), "position", {"rope", "none"});
    validate_heads(dim, b, path);
    return b;
}

}  // namespace

ModelConfig parse_model_config(const io::JsonValue& model_section,
                               std::int64_t vocab_size) {
    TITAN_CHECK(vocab_size >= 1,
                "parse_model_config: vocab_size must be >= 1, got " +
                    std::to_string(vocab_size));
    const io::JsonFields f =
        fields(model_section, "model",
               {"dim", "context_length", "n_layers", "block", "blocks",
                "rope_theta", "norm_eps", "tied_embeddings", "final_norm"});

    ModelConfig c;
    c.vocab_size = vocab_size;
    c.dim = f.integer(f.require("dim"), "dim", 1);
    c.context_length =
        f.integer(f.require("context_length"), "context_length", 1);
    if (const io::JsonValue* v = f.find("rope_theta")) {
        c.rope_theta = f.positive_float(*v, "rope_theta");
    }
    if (const io::JsonValue* v = f.find("norm_eps")) {
        c.norm_eps = f.positive_float(*v, "norm_eps");
    }
    if (const io::JsonValue* v = f.find("tied_embeddings")) {
        c.tied_embeddings = f.boolean(*v, "tied_embeddings");
    }
    if (const io::JsonValue* v = f.find("final_norm")) {
        c.final_norm = f.choice(*v, "final_norm", {"rmsnorm", "layernorm"});
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
            config_error(f.path_of("blocks"),
                         std::string("expected an array, got ") +
                             io::json_kind_name(*blocks));
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
        const std::int64_t n = f.integer(f.require("n_layers"), "n_layers", 1);
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
