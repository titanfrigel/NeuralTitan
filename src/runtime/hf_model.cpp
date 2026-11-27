#include "runtime/hf_model.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/tensor.hpp"
#include "io/file.hpp"
#include "io/json_fields.hpp"
#include "io/safetensors.hpp"
#include "nn/module.hpp"

namespace titan::runtime {

namespace {

constexpr std::string_view kEmbedding = "model.embed_tokens.weight";
constexpr std::string_view kFinalNorm = "model.norm.weight";
constexpr std::string_view kLmHead = "lm_head.weight";
constexpr std::string_view kLayerPrefix = "model.layers.";

// The tensors of one layer, model.layers.N.<hf> -> blocks.N.<titan>.
struct LayerTensor {
    std::string_view hf, titan;
};
constexpr std::array<LayerTensor, 9> kLayerTensors{{
    {"input_layernorm.weight", "norm1.weight"},
    {"post_attention_layernorm.weight", "norm2.weight"},
    {"self_attn.q_proj.weight", "attn.q_proj.weight"},
    {"self_attn.k_proj.weight", "attn.k_proj.weight"},
    {"self_attn.v_proj.weight", "attn.v_proj.weight"},
    {"self_attn.o_proj.weight", "attn.o_proj.weight"},
    {"mlp.gate_proj.weight", "mlp.gate_proj.weight"},
    {"mlp.up_proj.weight", "mlp.up_proj.weight"},
    {"mlp.down_proj.weight", "mlp.down_proj.weight"},
}};

// config.json files are a few hundred bytes; the cap only stops a huge file
// from being read into memory.
constexpr std::size_t kMaxConfigBytes = std::size_t{1} << 20;

// A key that is optional but supports one value only: when present, it must
// be that value. `expected` is the JSON text of the value, for the message.
void require_value(const io::JsonFields& f, std::string_view key, bool matches,
                   std::string_view expected) {
    if (f.find(key) != nullptr && !matches) {
        f.fail(f.path_of(key), "only " + std::string(expected) +
                                   " is supported, got " +
                                   io::to_json(*f.find(key), 0));
    }
}

// An optional bool that must be false when present.
void require_false(const io::JsonFields& f, std::string_view key) {
    if (const io::JsonValue* v = f.find(key)) {
        require_value(f, key, !f.boolean(*v, key), "false");
    }
}

// A token id: an integer in [0, vocab_size).
std::int32_t token_id(const io::JsonFields& f, std::string_view key,
                      std::int64_t vocab_size) {
    const std::int64_t id = f.integer(f.require(key), key, 0);
    if (id >= vocab_size) {
        f.fail(f.path_of(key), "must be below vocab_size " +
                                   std::to_string(vocab_size) + ", got " +
                                   std::to_string(id));
    }
    return static_cast<std::int32_t>(id);
}

// rope_theta (transformers 4.x) or rope_parameters (5.x), exactly one.
float rope_theta(const io::JsonFields& f) {
    const io::JsonValue* theta = f.find("rope_theta");
    const io::JsonValue* params = f.find("rope_parameters");
    if (theta != nullptr && params != nullptr) {
        f.fail(f.path_of("rope_parameters"),
               "cannot be combined with " + f.path_of("rope_theta"));
    }
    if (theta != nullptr) return f.positive_float(*theta, "rope_theta");
    if (params == nullptr) {
        f.fail(f.path_of("rope_theta"),
               "missing; give either rope_theta or rope_parameters");
    }
    const io::JsonFields p(*params, "config.json", f.path_of("rope_parameters"),
                           {"rope_type", "rope_theta"});
    // "default" is the plain RoPE of docs/math/rope.md; the other types
    // (linear, dynamic, yarn, llama3, ...) rescale the frequencies.
    p.choice(p.require("rope_type"), "rope_type", {"default"});
    return p.positive_float(p.require("rope_theta"), "rope_theta");
}

[[noreturn]] void fail_with(const std::string& context,
                            const std::string& message) {
    throw_error(__FILE__, __LINE__, context + ": " + message);
}

// The decimal layer index of "model.layers.<N>.<rest>", without sign or
// leading zeros, or -1 when `digits` is not one.
std::int64_t layer_index(std::string_view digits) {
    if (digits.empty() || digits.size() > 9 ||
        (digits.size() > 1 && digits[0] == '0')) {
        return -1;
    }
    std::int64_t n = 0;
    for (const char c : digits) {
        if (c < '0' || c > '9') return -1;
        n = 10 * n + (c - '0');
    }
    return n;
}

}  // namespace

HfLlamaConfig parse_hf_llama_config(const io::JsonValue& config) {
    const io::JsonFields f(
        config, "config.json", "",
        {// Required.
         "architectures", "model_type", "hidden_size", "intermediate_size",
         "num_attention_heads", "num_hidden_layers", "max_position_embeddings",
         "vocab_size", "rms_norm_eps", "bos_token_id", "eos_token_id",
         "rope_theta", "rope_parameters",
         // Optional, with Hugging Face's defaults.
         "num_key_value_heads", "tie_word_embeddings", "hidden_act",
         "attention_bias", "mlp_bias", "rope_scaling", "pretraining_tp",
         "rope_interleaved", "head_dim", "pad_token_id",
         // Informational: accepted with any value and ignored.
         "attention_dropout", "initializer_range", "torch_dtype", "dtype",
         "transformers_version", "use_cache", "is_llama_config",
         "_name_or_path"});

    const io::JsonValue& arch = f.require("architectures");
    const bool llama = arch.is_array() && arch.as_array().size() == 1 &&
                       arch.as_array()[0].is_string() &&
                       arch.as_array()[0].as_string() == "LlamaForCausalLM";
    if (!llama) {
        f.fail(f.path_of("architectures"),
               "expected [\"LlamaForCausalLM\"], got " + io::to_json(arch, 0));
    }
    f.choice(f.require("model_type"), "model_type", {"llama"});

    const std::int64_t hidden =
        f.integer(f.require("hidden_size"), "hidden_size", 1);
    const std::int64_t intermediate =
        f.integer(f.require("intermediate_size"), "intermediate_size", 1);
    const std::int64_t heads =
        f.integer(f.require("num_attention_heads"), "num_attention_heads", 1);
    const std::int64_t layers =
        f.integer(f.require("num_hidden_layers"), "num_hidden_layers", 1);
    if (layers > kMaxHfLayers) {
        f.fail(f.path_of("num_hidden_layers"),
               "must be at most " + std::to_string(kMaxHfLayers) + ", got " +
                   std::to_string(layers));
    }
    const std::int64_t context = f.integer(f.require("max_position_embeddings"),
                                           "max_position_embeddings", 1);
    if (context > kMaxHfContext) {
        f.fail(f.path_of("max_position_embeddings"),
               "must be at most " + std::to_string(kMaxHfContext) + ", got " +
                   std::to_string(context));
    }
    const std::int64_t vocab =
        f.integer(f.require("vocab_size"), "vocab_size", 1);
    // Token ids are int32 throughout.
    if (vocab > std::numeric_limits<std::int32_t>::max()) {
        f.fail(f.path_of("vocab_size"),
               "must be below 2^31, got " + std::to_string(vocab));
    }

    HfLlamaConfig out;
    out.bos_id = token_id(f, "bos_token_id", vocab);
    out.eos_id = token_id(f, "eos_token_id", vocab);

    std::int64_t kv_heads = heads;
    if (const io::JsonValue* v = f.find("num_key_value_heads")) {
        kv_heads = f.integer(*v, "num_key_value_heads", 1);
    }
    if (hidden % heads != 0) {
        f.fail(f.path_of("num_attention_heads"),
               "hidden_size " + std::to_string(hidden) +
                   " is not divisible by num_attention_heads " +
                   std::to_string(heads));
    }
    if (heads % kv_heads != 0) {
        f.fail(f.path_of("num_key_value_heads"),
               "num_attention_heads " + std::to_string(heads) +
                   " is not divisible by num_key_value_heads " +
                   std::to_string(kv_heads));
    }
    // RoPE rotates the pairs (i, i + head_dim/2) of each head.
    const std::int64_t head_dim = hidden / heads;
    if (head_dim % 2 != 0) {
        f.fail(f.path_of("num_attention_heads"),
               "RoPE needs an even head dimension, hidden_size / "
               "num_attention_heads = " +
                   std::to_string(head_dim));
    }
    // Our attention derives the head size from dim / n_heads.
    if (const io::JsonValue* v = f.find("head_dim")) {
        require_value(
            f, "head_dim", f.integer(*v, "head_dim", 1) == head_dim,
            std::to_string(head_dim) + " (hidden_size / num_attention_heads)");
    }

    // The supported architecture: SiLU-gated MLP, no biases, unscaled
    // half-split RoPE, no tensor-parallel slicing of the MLP.
    if (const io::JsonValue* v = f.find("hidden_act")) {
        f.choice(*v, "hidden_act", {"silu"});
    }
    require_false(f, "attention_bias");
    require_false(f, "mlp_bias");
    require_false(f, "rope_interleaved");
    require_value(
        f, "rope_scaling",
        f.find("rope_scaling") != nullptr && f.find("rope_scaling")->is_null(),
        "null");
    if (const io::JsonValue* v = f.find("pretraining_tp")) {
        require_value(f, "pretraining_tp",
                      f.integer(*v, "pretraining_tp", 1) == 1, "1");
    }
    if (const io::JsonValue* v = f.find("pad_token_id"); v && !v->is_null()) {
        f.integer(*v, "pad_token_id", std::numeric_limits<std::int64_t>::min());
    }

    model::ModelConfig& m = out.model;
    m.vocab_size = vocab;
    m.dim = hidden;
    m.context_length = context;
    m.rope_theta = rope_theta(f);
    m.norm_eps = f.positive_float(f.require("rms_norm_eps"), "rms_norm_eps");
    m.tied_embeddings = false;
    if (const io::JsonValue* v = f.find("tie_word_embeddings")) {
        m.tied_embeddings = f.boolean(*v, "tie_word_embeddings");
    }
    m.final_norm = "rmsnorm";
    m.blocks.assign(static_cast<std::size_t>(layers),
                    model::BlockConfig{heads, kv_heads, intermediate, "swiglu",
                                       "rmsnorm", "rope"});
    // The sizes together bound the model; name the size that scales it most.
    try {
        model::parameter_count(m);
    } catch (const Error& e) {
        f.fail(f.path_of("hidden_size"),
               std::string("the sizes imply too many parameters: ") + e.what());
    }
    return out;
}

std::string hf_to_titan_name(std::string_view hf_name, std::int64_t n_layers) {
    const std::string context =
        "Hugging Face tensor \"" + std::string(hf_name) + "\"";
    if (hf_name == kEmbedding) return "embed_tokens.weight";
    if (hf_name == kFinalNorm) return "final_norm.weight";
    if (hf_name == kLmHead) return "lm_head.weight";
    if (hf_name.starts_with(kLayerPrefix)) {
        const std::string_view rest = hf_name.substr(kLayerPrefix.size());
        const std::size_t dot = rest.find('.');
        const std::int64_t n = layer_index(rest.substr(0, dot));
        if (dot != std::string_view::npos && n >= 0) {
            const std::string_view leaf = rest.substr(dot + 1);
            for (const LayerTensor& t : kLayerTensors) {
                if (leaf != t.hf) continue;
                if (n >= n_layers) {
                    fail_with(context, "layer " + std::to_string(n) +
                                           " is out of range for " +
                                           std::to_string(n_layers) +
                                           " layers");
                }
                return "blocks." + std::to_string(n) + "." +
                       std::string(t.titan);
            }
        }
    }
    fail_with(context, "not a tensor of a Llama model");
}

HfModel load_hf_model(const std::filesystem::path& dir) {
    const std::string context = "hf model " + dir.string();
    const auto fail = [&](const std::string& message) {
        fail_with(context, message);
    };
    // Rethrows an error of a parser with the directory in front.
    const auto with_context = [&](auto&& fn) {
        try {
            return fn();
        } catch (const Error& e) {
            fail_with(context, e.what());
        }
    };

    // 1. config.json, before any other file is opened (see the header).
    HfModel out;
    out.config = with_context([&] {
        return parse_hf_llama_config(io::parse_json(
            io::read_file(dir / "config.json", kMaxConfigBytes)));
    });
    const model::ModelConfig& config = out.config.model;
    const std::int64_t n_layers =
        static_cast<std::int64_t>(config.blocks.size());

    // 2. tokenizer.json.
    out.tokenizer = with_context([&] {
        return tokenizer::load_tokenizer(
            io::parse_json(io::read_file(dir / "tokenizer.json")));
    });
    if (out.tokenizer->vocab_size() != config.vocab_size) {
        fail("tokenizer.json has " +
             std::to_string(out.tokenizer->vocab_size()) +
             " tokens but config.json: vocab_size is " +
             std::to_string(config.vocab_size));
    }
    const std::optional<std::int32_t> eos = out.tokenizer->eos_id();
    if (eos != out.config.eos_id) {
        fail("tokenizer.json's end-of-text token is " +
             (eos ? std::to_string(*eos) : std::string("absent")) +
             " but config.json: eos_token_id is " +
             std::to_string(out.config.eos_id));
    }

    // 3. The safetensors header: names and element counts.
    const std::filesystem::path weights_path = dir / "model.safetensors";
    const io::SafetensorsFile weights =
        with_context([&] { return io::SafetensorsFile::open(weights_path); });
    const std::string weights_context = weights_path.string();
    const std::string tied_note = std::string(" with tie_word_embeddings ") +
                                  (config.tied_embeddings ? "true" : "false");

    // Our parameter name -> its tensor.
    std::map<std::string, const io::SafetensorsEntry*> by_param;
    std::int64_t stored = 0;
    for (const io::SafetensorsEntry& e : weights.entries()) {
        if (e.name == kLmHead && config.tied_embeddings) {
            fail(weights_context + ": unexpected tensor lm_head.weight" +
                 tied_note + " (the head reuses model.embed_tokens.weight)");
        }
        std::string name;
        try {
            name = hf_to_titan_name(e.name, n_layers);
        } catch (const Error& err) {
            fail(weights_context + ": unexpected tensor: " + err.what());
        }
        const auto [it, inserted] = by_param.emplace(name, &e);
        if (!inserted) {
            fail(weights_context + ": tensors " + it->second->name + " and " +
                 e.name + " both map to " + name);
        }
        std::int64_t n = 1;
        for (const std::int64_t d : e.shape) n *= d;
        stored += n;  // bounded by the file size, see SafetensorsFile::open
    }
    // Every expected tensor, in Hugging Face's names.
    std::vector<std::string> expected{std::string(kEmbedding),
                                      std::string(kFinalNorm)};
    for (std::int64_t i = 0; i < n_layers; ++i) {
        for (const LayerTensor& t : kLayerTensors) {
            expected.push_back(std::string(kLayerPrefix) + std::to_string(i) +
                               "." + std::string(t.hf));
        }
    }
    if (!config.tied_embeddings) expected.emplace_back(kLmHead);
    for (const std::string& hf : expected) {
        if (weights.find(hf) == nullptr) {
            fail(weights_context + ": missing tensor " + hf +
                 (hf == kLmHead ? tied_note : std::string()));
        }
    }
    // Names now map one to one, so the counts differ only through shapes. The
    // check bounds the model by the file before it is allocated.
    const std::int64_t params = model::parameter_count(config);
    if (params != stored) {
        fail(weights_context + ": config.json implies " +
             std::to_string(params) + " parameters, but the tensors hold " +
             std::to_string(stored) + " values");
    }

    // 4. The model, and every shape. Weights are (out, in) on both sides.
    out.model = std::make_unique<model::Model>(config);
    const std::vector<nn::Parameter*> parameters = out.model->parameters();
    TITAN_CHECK(parameters.size() == by_param.size(),
                context + ": the model has " +
                    std::to_string(parameters.size()) + " parameters but " +
                    std::to_string(by_param.size()) + " tensors map to them");
    std::vector<const io::SafetensorsEntry*> sources;
    for (const nn::Parameter* p : parameters) {
        const auto it = by_param.find(p->name);
        TITAN_CHECK(it != by_param.end(),
                    context + ": no tensor maps to parameter " + p->name);
        const io::SafetensorsEntry& e = *it->second;
        if (e.shape != p->value.shape()) {
            fail(weights_context + ": tensor " + e.name + " has shape " +
                 to_string(e.shape) + " but " + p->name + " has shape " +
                 to_string(p->value.shape()));
        }
        sources.push_back(&e);
    }

    // 5. The values, widened to FP32.
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const std::vector<float> values =
            with_context([&] { return weights.read_f32(*sources[i]); });
        parameters[i]->value = Tensor::from_vector(
            values, parameters[i]->value.shape(), Device::CPU);
    }
    return out;
}

}  // namespace titan::runtime
