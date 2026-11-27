#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "io/safetensors.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "nn/module.hpp"
#include "runtime/hf_model.hpp"
#include "support/safetensors_writer.hpp"

namespace {

using titan::io::JsonValue;
using titan::runtime::hf_to_titan_name;
using titan::runtime::HfLlamaConfig;
using titan::runtime::load_hf_model;
using titan::runtime::parse_hf_llama_config;

// SmolLM2-135M's config.json at revision 93efa2f (transformers 4.40.1).
constexpr const char* kSmolLM2Config = R"({
  "architectures": ["LlamaForCausalLM"], "attention_bias": false,
  "attention_dropout": 0.0, "bos_token_id": 0, "eos_token_id": 0,
  "hidden_act": "silu", "hidden_size": 576,
  "initializer_range": 0.041666666666666664, "intermediate_size": 1536,
  "is_llama_config": true, "max_position_embeddings": 8192,
  "model_type": "llama", "num_attention_heads": 9, "num_hidden_layers": 30,
  "num_key_value_heads": 3, "pretraining_tp": 1, "rms_norm_eps": 1e-05,
  "rope_interleaved": false, "rope_scaling": null, "rope_theta": 100000,
  "tie_word_embeddings": true, "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1", "use_cache": true, "vocab_size": 49152
})";

JsonValue smollm2_config() { return titan::io::parse_json(kSmolLM2Config); }

titan::model::ModelConfig smollm2_model_config() {
    titan::model::ModelConfig c;
    c.vocab_size = 49152;
    c.dim = 576;
    c.context_length = 8192;
    c.rope_theta = 100000.f;
    c.norm_eps = 1e-5f;
    c.tied_embeddings = true;
    c.final_norm = "rmsnorm";
    c.blocks.assign(
        30, titan::model::BlockConfig{9, 3, 1536, "swiglu", "rmsnorm", "rope"});
    return c;
}

// A copy of `object` without `key`.
JsonValue without(const JsonValue& object, const std::string& key) {
    JsonValue out = JsonValue::object();
    for (const auto& [k, v] : object.as_object()) {
        if (k != key) out.set(k, v);
    }
    return out;
}

// Whether `fn` throws a titan::Error whose message holds every needle.
bool throws_with(const std::function<void()>& fn,
                 std::initializer_list<std::string> needles) {
    try {
        fn();
    } catch (const titan::Error& e) {
        const std::string message = e.what();
        bool ok = true;
        for (const std::string& n : needles) {
            ok = ok && message.find(n) != std::string::npos;
        }
        return ok;
    }
    return false;
}

// Whether parsing the SmolLM2 config after `edit` fails naming
// "config.json: <field>".
bool config_rejects(const std::function<void(JsonValue&)>& edit,
                    const std::string& field) {
    JsonValue c = smollm2_config();
    edit(c);
    return throws_with([&] { parse_hf_llama_config(c); },
                       {"config.json: " + field});
}

JsonValue rope_parameters(const char* type, double theta) {
    JsonValue p = JsonValue::object();
    p.set("rope_type", JsonValue::string(type));
    p.set("rope_theta", JsonValue::number(theta));
    return p;
}

}  // namespace

TITAN_TEST(HfConfig, ParsesSmolLM2) {
    const HfLlamaConfig c = parse_hf_llama_config(smollm2_config());
    TITAN_EXPECT(c.model == smollm2_model_config());
    TITAN_EXPECT_EQ(c.bos_id, 0);
    TITAN_EXPECT_EQ(c.eos_id, 0);
}

TITAN_TEST(HfConfig, ParsesRopeParameters) {
    JsonValue j = without(smollm2_config(), "rope_theta");
    j.set("rope_parameters", rope_parameters("default", 100000.0));
    const HfLlamaConfig c = parse_hf_llama_config(j);
    TITAN_EXPECT(c.model == smollm2_model_config());
    TITAN_EXPECT_EQ(c.bos_id, 0);
    TITAN_EXPECT_EQ(c.eos_id, 0);
}

TITAN_TEST(HfConfig, Defaults) {
    // Only the required keys: num_key_value_heads defaults to
    // num_attention_heads (no GQA) and tie_word_embeddings to false.
    const JsonValue j = titan::io::parse_json(R"({
      "architectures": ["LlamaForCausalLM"], "model_type": "llama",
      "hidden_size": 64, "intermediate_size": 160, "num_attention_heads": 4,
      "num_hidden_layers": 2, "max_position_embeddings": 128,
      "vocab_size": 512, "rms_norm_eps": 1e-6, "bos_token_id": 1,
      "eos_token_id": 2, "rope_theta": 500000.0
    })");
    const HfLlamaConfig c = parse_hf_llama_config(j);
    titan::model::ModelConfig want;
    want.vocab_size = 512;
    want.dim = 64;
    want.context_length = 128;
    want.rope_theta = 500000.f;
    want.norm_eps = 1e-6f;
    want.tied_embeddings = false;
    want.final_norm = "rmsnorm";
    want.blocks.assign(
        2, titan::model::BlockConfig{4, 4, 160, "swiglu", "rmsnorm", "rope"});
    TITAN_EXPECT(c.model == want);
    TITAN_EXPECT_EQ(c.bos_id, 1);
    TITAN_EXPECT_EQ(c.eos_id, 2);
}

TITAN_TEST(HfConfig, RejectsUnknownKey) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) { c.set("sliding_window", JsonValue::number(4096)); },
        "sliding_window: unknown key"));
}

TITAN_TEST(HfConfig, RejectsRopeScaling) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) {
            JsonValue s = JsonValue::object();
            s.set("type", JsonValue::string("linear"));
            c.set("rope_scaling", s);
        },
        "rope_scaling"));
}

TITAN_TEST(HfConfig, RejectsAttentionBias) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) { c.set("attention_bias", JsonValue::boolean(true)); },
        "attention_bias"));
}

TITAN_TEST(HfConfig, RejectsHiddenAct) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) { c.set("hidden_act", JsonValue::string("gelu")); },
        "hidden_act"));
}

TITAN_TEST(HfConfig, RejectsHeadDim) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) { c.set("head_dim", JsonValue::number(32)); },
        "head_dim"));
}

TITAN_TEST(HfConfig, RejectsRopeInterleaved) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) {
            c.set("rope_interleaved", JsonValue::boolean(true));
        },
        "rope_interleaved"));
}

TITAN_TEST(HfConfig, RejectsArchitecture) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) {
            JsonValue a = JsonValue::array();
            a.push_back(JsonValue::string("MistralForCausalLM"));
            c.set("architectures", a);
        },
        "architectures"));
}

TITAN_TEST(HfConfig, RejectsBothRopeForms) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) {
            c.set("rope_parameters", rope_parameters("default", 100000.0));
        },
        "rope_parameters"));
}

TITAN_TEST(HfConfig, RejectsRopeType) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) {
            c = without(c, "rope_theta");
            c.set("rope_parameters", rope_parameters("llama3", 100000.0));
        },
        "rope_parameters.rope_type"));
}

TITAN_TEST(HfConfig, RejectsHeadsNotDividingHidden) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) {
            c.set("num_attention_heads", JsonValue::number(7));
        },
        "num_attention_heads"));
}

TITAN_TEST(HfConfig, RejectsEosTokenList) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) {
            JsonValue ids = JsonValue::array();
            ids.push_back(JsonValue::number(0));
            ids.push_back(JsonValue::number(2));
            c.set("eos_token_id", ids);
        },
        "eos_token_id"));
}

TITAN_TEST(HfConfig, RejectsTokenIdOutOfRange) {
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) { c.set("bos_token_id", JsonValue::number(49152)); },
        "bos_token_id: must be below vocab_size 49152"));
}

TITAN_TEST(HfConfig, RejectsHugeModel) {
    // Too many layers for the block list, and sizes whose parameter count
    // passes 2^60, both fail before anything is allocated.
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) {
            c.set("num_hidden_layers", JsonValue::number(1e15));
        },
        "num_hidden_layers: must be at most 4096"));
    TITAN_EXPECT(config_rejects(
        [](JsonValue& c) {
            c.set("hidden_size", JsonValue::number(9 * 64 * 1e9));
        },
        "hidden_size: the sizes imply too many parameters"));
}

TITAN_TEST(HfName, Maps) {
    const auto map = [](const char* name) { return hf_to_titan_name(name, 2); };
    TITAN_EXPECT_EQ(map("model.embed_tokens.weight"),
                    std::string("embed_tokens.weight"));
    TITAN_EXPECT_EQ(map("model.layers.0.input_layernorm.weight"),
                    std::string("blocks.0.norm1.weight"));
    TITAN_EXPECT_EQ(map("model.layers.1.post_attention_layernorm.weight"),
                    std::string("blocks.1.norm2.weight"));
    for (const char* p : {"q", "k", "v", "o"}) {
        TITAN_EXPECT_EQ(
            map(("model.layers.1.self_attn." + std::string(p) + "_proj.weight")
                    .c_str()),
            "blocks.1.attn." + std::string(p) + "_proj.weight");
    }
    for (const char* p : {"gate", "up", "down"}) {
        TITAN_EXPECT_EQ(
            map(("model.layers.0.mlp." + std::string(p) + "_proj.weight")
                    .c_str()),
            "blocks.0.mlp." + std::string(p) + "_proj.weight");
    }
    TITAN_EXPECT_EQ(map("model.norm.weight"), std::string("final_norm.weight"));
    TITAN_EXPECT_EQ(map("lm_head.weight"), std::string("lm_head.weight"));
    TITAN_EXPECT_EQ(hf_to_titan_name("model.layers.29.mlp.up_proj.weight", 30),
                    std::string("blocks.29.mlp.up_proj.weight"));

    for (const char* bad :
         {"model.layers.2.self_attn.q_proj.weight",
          "model.layers.0.self_attn.rotary_emb.inv_freq",
          "model.layers.01.mlp.up_proj.weight", "model.layers.-1.mlp.up_proj",
          "model.layers..mlp.up_proj.weight", "model.lm_head.weight",
          "model.norm.bias", ""}) {
        TITAN_EXPECT(throws_with([&] { map(bad); }, {bad}));
    }
}

namespace {

const std::filesystem::path kTinyDir =
    std::filesystem::path(TITAN_FIXTURES_DIR) / "tiny_llama";

std::int64_t numel(const std::vector<std::int64_t>& shape) {
    std::int64_t n = 1;
    for (std::int64_t d : shape) n *= d;
    return n;
}

// A tiny fixture's tensors with their raw bytes, in file order of the
// entries (sorted by name), for rewriting with write_safetensors.
std::vector<titan::test::TestTensor> raw_tensors(
    const std::filesystem::path& file) {
    const auto st = titan::io::SafetensorsFile::open(file);
    const std::string bytes = titan::io::read_file(file);
    std::uint64_t header = 0;
    std::memcpy(&header, bytes.data(), sizeof header);  // little-endian host
    const std::size_t data = 8 + static_cast<std::size_t>(header);
    std::vector<titan::test::TestTensor> out;
    for (const auto& e : st.entries()) {
        out.push_back({e.name, "BF16", e.shape,
                       bytes.substr(data + e.begin, e.end - e.begin)});
    }
    return out;
}

// A fresh copy of tiny_llama/<variant> under the temp directory.
std::filesystem::path copy_tiny(const char* variant, const char* name) {
    const auto dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    for (const char* f :
         {"config.json", "tokenizer.json", "model.safetensors"}) {
        std::filesystem::copy_file(kTinyDir / variant / f, dir / f);
    }
    return dir;
}

// Rewrites dir/model.safetensors after `edit` changes its tensor list.
void edit_tensors(
    const std::filesystem::path& dir,
    const std::function<void(std::vector<titan::test::TestTensor>&)>& edit) {
    auto tensors = raw_tensors(dir / "model.safetensors");
    edit(tensors);
    titan::test::write_safetensors(dir / "model.safetensors", tensors);
}

void erase_tensor(std::vector<titan::test::TestTensor>& tensors,
                  const std::string& name) {
    std::erase_if(tensors, [&](const auto& t) { return t.name == name; });
}

// Whether loading `dir` fails with every needle in the message.
bool load_rejects(const std::filesystem::path& dir,
                  std::initializer_list<std::string> needles) {
    const bool ok = throws_with([&] { load_hf_model(dir); }, needles);
    std::filesystem::remove_all(dir);
    return ok;
}

}  // namespace

TITAN_TEST(HfModel, LoadsTiny) {
    for (const char* variant : {"tied", "untied"}) {
        const auto dir = kTinyDir / variant;
        const titan::runtime::HfModel m = load_hf_model(dir);
        const auto st =
            titan::io::SafetensorsFile::open(dir / "model.safetensors");
        std::int64_t stored = 0;
        for (const auto& e : st.entries()) stored += numel(e.shape);
        TITAN_EXPECT_EQ(m.model->num_parameters(), stored);
        TITAN_EXPECT_EQ(m.config.model.tied_embeddings,
                        std::string(variant) == "tied");
        TITAN_EXPECT_EQ(m.tokenizer->vocab_size(), std::int64_t{512});
        TITAN_EXPECT_EQ(m.config.eos_id, 0);

        // One sampled weight, element (3, 5) of layer 1's k_proj, equals the
        // BF16 bits at its offset widened to FP32.
        const auto* entry = st.find("model.layers.1.self_attn.k_proj.weight");
        TITAN_EXPECT(entry != nullptr);
        const std::int64_t index = 3 * entry->shape[1] + 5;
        const auto raw = raw_tensors(dir / "model.safetensors");
        std::uint16_t bits = 0;
        for (const auto& t : raw) {
            if (t.name == entry->name) {
                std::memcpy(&bits, t.bytes.data() + 2 * index, sizeof bits);
            }
        }
        for (titan::nn::Parameter* p : m.model->parameters()) {
            if (p->name == "blocks.1.attn.k_proj.weight") {
                TITAN_EXPECT(p->value.shape() == entry->shape);
                TITAN_EXPECT_EQ(p->value.data()[index],
                                titan::io::bf16_to_f32(bits));
                TITAN_EXPECT(p->value.data()[index] != 0.0f);
            }
        }
    }
}

TITAN_TEST(HfModel, RejectsMissingTensor) {
    const auto dir = copy_tiny("tied", "titan_hf_missing");
    edit_tensors(dir, [](auto& t) {
        erase_tensor(t, "model.layers.1.mlp.up_proj.weight");
    });
    TITAN_EXPECT(load_rejects(dir, {"model.safetensors", "missing tensor",
                                    "model.layers.1.mlp.up_proj.weight"}));
}

TITAN_TEST(HfModel, RejectsExtraTensor) {
    const auto dir = copy_tiny("tied", "titan_hf_extra");
    edit_tensors(dir, [](auto& t) {
        t.push_back({"model.layers.0.self_attn.rotary_emb.inv_freq",
                     "BF16",
                     {8},
                     std::string(16, '\0')});
    });
    TITAN_EXPECT(load_rejects(
        dir,
        {"model.safetensors", "model.layers.0.self_attn.rotary_emb.inv_freq"}));
}

TITAN_TEST(HfModel, RejectsTransposedShape) {
    // k_proj, since the tiny q_proj is square: [64, 32] instead of [32, 64].
    const auto dir = copy_tiny("tied", "titan_hf_transposed");
    edit_tensors(dir, [](auto& tensors) {
        for (auto& t : tensors) {
            if (t.name == "model.layers.0.self_attn.k_proj.weight") {
                t.shape = {64, 32};
            }
        }
    });
    TITAN_EXPECT(load_rejects(dir, {"model.layers.0.self_attn.k_proj.weight",
                                    "[64, 32]", "[32, 64]"}));
}

TITAN_TEST(HfModel, RejectsLmHeadWithTiedEmbeddings) {
    const auto dir = copy_tiny("tied", "titan_hf_tied_head");
    edit_tensors(dir, [](auto& t) {
        for (const auto& x : std::vector(t)) {
            if (x.name == "model.embed_tokens.weight") {
                t.push_back({"lm_head.weight", x.dtype, x.shape, x.bytes});
            }
        }
    });
    TITAN_EXPECT(load_rejects(dir, {"lm_head.weight", "tie_word_embeddings"}));
}

TITAN_TEST(HfModel, RejectsMissingLmHeadUntied) {
    const auto dir = copy_tiny("untied", "titan_hf_untied_head");
    edit_tensors(dir, [](auto& t) { erase_tensor(t, "lm_head.weight"); });
    TITAN_EXPECT(load_rejects(dir, {"lm_head.weight", "tie_word_embeddings"}));
}

TITAN_TEST(HfModel, RejectsTokenizerVocabulary) {
    // SmolLM2's tokenizer has 49152 tokens, the tiny config 512.
    const auto dir = copy_tiny("tied", "titan_hf_vocab");
    std::filesystem::copy_file(
        std::filesystem::path(TITAN_FIXTURES_DIR) / "smollm2" /
            "tokenizer.json",
        dir / "tokenizer.json",
        std::filesystem::copy_options::overwrite_existing);
    TITAN_EXPECT(load_rejects(dir, {"tokenizer.json", "49152", "512"}));
}

TITAN_TEST(HfModel, RejectsEosTokenId) {
    const auto dir = copy_tiny("tied", "titan_hf_eos");
    JsonValue c =
        titan::io::parse_json(titan::io::read_file(dir / "config.json"));
    c.set("eos_token_id", JsonValue::number(1));
    titan::io::write_file(dir / "config.json", titan::io::to_json(c));
    TITAN_EXPECT(load_rejects(dir, {"tokenizer.json", "eos_token_id"}));
}

TITAN_TEST(HfModel, RejectsMissingConfig) {
    const auto dir = copy_tiny("tied", "titan_hf_no_config");
    std::filesystem::remove(dir / "config.json");
    TITAN_EXPECT(load_rejects(dir, {"config.json"}));
}
