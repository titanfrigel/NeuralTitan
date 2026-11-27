#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "io/json.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "tokenizer/tokenizer.hpp"

namespace titan::runtime {

// A Hugging Face Llama config.json, as our ModelConfig plus the special token
// ids. See docs/math/hf-llama.md.
struct HfLlamaConfig {
    model::ModelConfig model;
    std::int32_t bos_id{}, eos_id{};
};

// The most layers parse_hf_llama_config accepts, far above any released Llama
// (Llama 3.1 405B has 126). It keeps the block list small before
// parameter_count, which only bounds the parameters, is consulted.
inline constexpr std::int64_t kMaxHfLayers = 4096;

// The longest max_position_embeddings parse_hf_llama_config accepts. The KV
// cache allocates context_length rows per layer, so an unbounded context in
// an untrusted config could ask for any amount of memory; SmolLM2 uses 8192.
inline constexpr std::int64_t kMaxHfContext = std::int64_t{1} << 20;

// Parses and validates a Hugging Face LlamaForCausalLM config.json (the
// format of transformers 4.x and 5.x). Required:
//   architectures ["LlamaForCausalLM"], model_type "llama", hidden_size,
//   intermediate_size, num_attention_heads, num_hidden_layers (at most
//   kMaxHfLayers), max_position_embeddings (at most kMaxHfContext), vocab_size,
//   rms_norm_eps, and bos_token_id and eos_token_id (integers in [0,
//   vocab_size)), plus exactly one of rope_theta (4.x) or rope_parameters
//   {"rope_type": "default", "rope_theta": ...} (5.x).
// Optional, with Hugging Face's defaults: num_key_value_heads
// (num_attention_heads), tie_word_embeddings (false), and keys that only one
// value of is supported: hidden_act "silu", attention_bias and mlp_bias false,
// rope_scaling null, pretraining_tp 1, rope_interleaved false, head_dim
// hidden_size / num_attention_heads, pad_token_id null or an integer.
// Accepted with any value and ignored: attention_dropout, initializer_range,
// torch_dtype, dtype, transformers_version, use_cache, is_llama_config and
// _name_or_path. Any other key is rejected.
//
// The result is n identical blocks of SwiGLU (mlp_hidden =
// intermediate_size), RMSNorm and RoPE, an RMSNorm final norm, context_length
// = max_position_embeddings and norm_eps = rms_norm_eps. Its parameter_count
// is checked against kMaxParameters before returning.
//
// Throws titan::Error naming the field path, e.g. "config.json:
// rope_scaling", for a missing, unknown or unsupported key, a wrong type, a
// non-positive size, hidden_size % num_attention_heads != 0,
// num_attention_heads % num_key_value_heads != 0 or an odd head dimension.
HfLlamaConfig parse_hf_llama_config(const io::JsonValue& config);

// Our parameter name for a Hugging Face Llama tensor name:
//   model.embed_tokens.weight -> embed_tokens.weight
//   model.layers.N. + input_layernorm.weight -> blocks.N.norm1.weight
//                     post_attention_layernorm.weight -> blocks.N.norm2.weight
//                     self_attn.{q,k,v,o}_proj.weight -> blocks.N.attn.<same>
//                     mlp.{gate,up,down}_proj.weight -> blocks.N.mlp.<same>
//   model.norm.weight -> final_norm.weight
//   lm_head.weight -> lm_head.weight (the untied head)
// N is written in decimal without leading zeros. Throws titan::Error naming
// the tensor for any other name, or for N >= n_layers.
std::string hf_to_titan_name(std::string_view hf_name, std::int64_t n_layers);

// A Hugging Face model loaded into our Model, on the CPU.
struct HfModel {
    HfLlamaConfig config;
    std::unique_ptr<model::Model> model;
    std::unique_ptr<tokenizer::Tokenizer> tokenizer;
};

// Loads a Hugging Face Llama directory: config.json, tokenizer.json and
// model.safetensors. The files are untrusted and are read in this order:
//
// 1. config.json is parsed and validated (parse_hf_llama_config) before any
//    other file is opened, so a bad config fails the same way whether or not
//    the other files exist.
// 2. tokenizer.json is loaded with tokenizer::load_tokenizer. Its vocabulary
//    size must equal vocab_size and its end-of-text id eos_token_id.
// 3. model.safetensors's header is read (io::SafetensorsFile::open). Every
//    tensor name must map (hf_to_titan_name) to a distinct parameter, and
//    every parameter must have a tensor: lm_head.weight is rejected with tied
//    embeddings and required without them. The tensors' element counts must
//    add up to the config's parameter_count, which bounds the model by the
//    file size before the model is allocated.
// 4. The model is built, and every tensor's shape must equal its parameter's.
// 5. Only then are the tensors read, converted to FP32 and stored in the
//    parameters.
//
// Throws titan::Error naming the directory and the file, and the field or
// tensor (with both shapes for a shape mismatch).
HfModel load_hf_model(const std::filesystem::path& dir);

}  // namespace titan::runtime
