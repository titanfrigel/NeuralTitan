#include "runtime/checkpoint.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "io/checkpoint_file.hpp"
#include "io/json_fields.hpp"
#include "nn/module.hpp"

namespace titan::runtime {

namespace {

constexpr std::string_view kMomentM = "optim.m.";
constexpr std::string_view kMomentV = "optim.v.";

std::string file_context(const std::filesystem::path& path) {
    return "checkpoint " + path.string();
}

// The "state" object of the meta, see checkpoint.hpp.
io::JsonValue state_to_json(const TrainerState& s) {
    io::JsonValue o = io::JsonValue::object();
    o.set("step", io::JsonValue::number(static_cast<double>(s.step)));
    o.set("optimizer_steps",
          io::JsonValue::number(static_cast<double>(s.optimizer.steps)));
    o.set("train_rng", io::JsonValue::string(s.train_rng));
    o.set("val_rng", io::JsonValue::string(s.val_rng));
    o.set("train_tokens",
          io::JsonValue::number(static_cast<double>(s.train_tokens)));
    o.set("val_tokens",
          io::JsonValue::number(static_cast<double>(s.val_tokens)));
    o.set("train_crc32",
          io::JsonValue::number(static_cast<double>(s.train_crc32)));
    o.set("val_crc32", io::JsonValue::number(static_cast<double>(s.val_crc32)));
    return o;
}

// A generator state string, checked by restoring it into a scratch Rng so a
// malformed one fails here, naming the field, rather than at resume.
std::string rng_state(const io::JsonFields& f, std::string_view key) {
    const io::JsonValue& v = f.require(key);
    if (!v.is_string()) {
        f.fail(f.path_of(key),
               std::string("expected a string, got ") + io::json_kind_name(v));
    }
    Rng probe(0);
    try {
        probe.load_state(v.as_string());
    } catch (const Error&) {
        f.fail(f.path_of(key), "malformed generator state");
    }
    return v.as_string();
}

// A CRC-32 value: an integer in [0, 2^32).
std::uint32_t crc32_field(const io::JsonFields& f, std::string_view key) {
    const std::int64_t v = f.integer(f.require(key), key, 0);
    if (v > std::numeric_limits<std::uint32_t>::max()) {
        f.fail(f.path_of(key), "must be below 2^32, got " + std::to_string(v));
    }
    return static_cast<std::uint32_t>(v);
}

TrainerState state_from_json(const io::JsonValue& value,
                             const std::string& context) {
    const io::JsonFields f(
        value, context, "meta.state",
        {"step", "optimizer_steps", "train_rng", "val_rng", "train_tokens",
         "val_tokens", "train_crc32", "val_crc32"});
    TrainerState s;
    s.step = f.integer(f.require("step"), "step", 0);
    s.optimizer.steps =
        f.integer(f.require("optimizer_steps"), "optimizer_steps", 0);
    s.train_rng = rng_state(f, "train_rng");
    s.val_rng = rng_state(f, "val_rng");
    s.train_tokens = f.integer(f.require("train_tokens"), "train_tokens", 0);
    s.val_tokens = f.integer(f.require("val_tokens"), "val_tokens", 0);
    s.train_crc32 = crc32_field(f, "train_crc32");
    s.val_crc32 = crc32_field(f, "val_crc32");
    return s;
}

// Everything of a checkpoint but the optimizer moments, from the meta object.
Checkpoint checkpoint_from_meta(const io::JsonValue& meta,
                                const std::string& context) {
    const io::JsonFields f(
        meta, context, "meta",
        {"vocab_size", "model", "train", "tokenizer", "state"});
    const std::int64_t vocab_size =
        f.integer(f.require("vocab_size"), "vocab_size", 1);
    Checkpoint c;
    // The section parsers name their own paths ("model.dim", "train.lr");
    // prefix the file so the message says which checkpoint is at fault.
    try {
        c.model = model::parse_model_config(f.require("model"), vocab_size);
        c.train = parse_train_config(f.require("train"));
    } catch (const Error& e) {
        throw_error(__FILE__, __LINE__, context + ": " + e.what());
    }
    c.tokenizer = f.require("tokenizer");
    c.state = state_from_json(f.require("state"), context);
    return c;
}

// Where two JSON values first differ: the dotted path below `path` and both
// values as compact JSON, or an empty path when they are equal.
struct Difference {
    std::string path, in_file, in_model;
};

Difference first_difference(const io::JsonValue& a, const io::JsonValue& b,
                            const std::string& path) {
    if (a.is_object() && b.is_object()) {
        for (const auto& [key, va] : a.as_object()) {
            const io::JsonValue* vb = b.find(key);
            if (vb == nullptr)
                return {path + "." + key, io::to_json(va, 0), "absent"};
            Difference d = first_difference(va, *vb, path + "." + key);
            if (!d.path.empty()) return d;
        }
        for (const auto& [key, vb] : b.as_object()) {
            if (a.find(key) == nullptr) {
                return {path + "." + key, "absent", io::to_json(vb, 0)};
            }
        }
        return {};
    }
    if (a.is_array() && b.is_array() &&
        a.as_array().size() == b.as_array().size()) {
        for (std::size_t i = 0; i < a.as_array().size(); ++i) {
            Difference d =
                first_difference(a.as_array()[i], b.as_array()[i],
                                 path + "[" + std::to_string(i) + "]");
            if (!d.path.empty()) return d;
        }
        return {};
    }
    const std::string ta = io::to_json(a, 0), tb = io::to_json(b, 0);
    if (ta == tb) return {};
    return {path, ta, tb};
}

// Throws unless the checkpoint's model config equals the model's, naming the
// first field that differs.
void check_same_config(const model::ModelConfig& in_file,
                       const model::ModelConfig& in_model,
                       const std::string& context) {
    if (in_file == in_model) return;
    TITAN_CHECK(in_file.vocab_size == in_model.vocab_size,
                context + ": meta.vocab_size is " +
                    std::to_string(in_file.vocab_size) +
                    " in the checkpoint but " +
                    std::to_string(in_model.vocab_size) + " in the model");
    const Difference d =
        first_difference(model::model_config_to_json(in_file),
                         model::model_config_to_json(in_model), "model");
    TITAN_CHECK(!d.path.empty(),
                context + ": the model config differs from the model's");
    throw_error(__FILE__, __LINE__,
                context + ": " + d.path + " is " + d.in_file +
                    " in the checkpoint but " + d.in_model + " in the model");
}

// Bounds the saved model config by the file before anyone builds a model
// from it: the parameters it implies, times three for the two AdamW moments,
// must equal the values the file's tensor entries hold. The entries are
// already bounded by the file size, so a config that claims a huge model
// fails here instead of in an allocation.
void check_sizes(const model::ModelConfig& config,
                 const std::vector<io::TensorInfo>& tensors,
                 const std::string& context) {
    std::int64_t stored = 0;
    for (const io::TensorInfo& t : tensors) {
        std::int64_t n = 1;
        for (const std::int64_t dim : t.shape) n *= dim;
        stored += n;
    }
    std::int64_t params = 0;
    try {
        params = model::parameter_count(config);
    } catch (const Error& e) {
        throw_error(__FILE__, __LINE__,
                    context + ": " + e.what() + ", but the file holds " +
                        std::to_string(stored) + " values");
    }
    TITAN_CHECK(3 * params == stored,
                context + ": the model config implies " +
                    std::to_string(params) + " parameters (" +
                    std::to_string(3 * params) +
                    " values with the optimizer moments), but the file holds " +
                    std::to_string(stored) + " values");
}

std::string shape_mismatch(const std::string& context, const std::string& name,
                           const Shape& in_file, const Shape& in_model) {
    return context + ": tensor " + name + " has shape " + to_string(in_file) +
           " in the checkpoint but " + to_string(in_model) + " in the model";
}

}  // namespace

void save_checkpoint(const std::filesystem::path& path, model::Model& model,
                     const Checkpoint& ckpt) {
    const std::vector<nn::Parameter*> params = model.parameters();
    const optim::AdamWState& opt = ckpt.state.optimizer;
    const std::string context = "save_checkpoint " + path.string();
    TITAN_CHECK(opt.m.size() == params.size() && opt.v.size() == params.size(),
                context + ": optimizer state has " +
                    std::to_string(opt.m.size()) + " first and " +
                    std::to_string(opt.v.size()) + " second moments for " +
                    std::to_string(params.size()) + " parameters");

    // The parameters, then every first moment, then every second moment.
    std::vector<io::NamedTensor> tensors;
    tensors.reserve(3 * params.size());
    for (const nn::Parameter* p : params) {
        tensors.push_back({p->name, p->value.to(Device::CPU)});
    }
    for (const auto& [prefix, moments] :
         {std::pair{kMomentM, &opt.m}, std::pair{kMomentV, &opt.v}}) {
        for (std::size_t i = 0; i < params.size(); ++i) {
            const Tensor& t = (*moments)[i];
            const std::string name = std::string(prefix) + params[i]->name;
            TITAN_CHECK(t.shape() == params[i]->value.shape(),
                        shape_mismatch(context, name, t.shape(),
                                       params[i]->value.shape()));
            tensors.push_back({name, t.to(Device::CPU)});
        }
    }

    io::JsonValue meta = io::JsonValue::object();
    meta.set("vocab_size",
             io::JsonValue::number(static_cast<double>(ckpt.model.vocab_size)));
    meta.set("model", model::model_config_to_json(ckpt.model));
    meta.set("train", train_config_to_json(ckpt.train));
    meta.set("tokenizer", ckpt.tokenizer);
    meta.set("state", state_to_json(ckpt.state));
    io::write_checkpoint_file(path, meta, tensors);
}

Checkpoint load_checkpoint(const std::filesystem::path& path,
                           model::Model* model) {
    const std::string context = file_context(path);
    if (model == nullptr) {
        const io::CheckpointHeader header = io::read_checkpoint_header(path);
        Checkpoint c = checkpoint_from_meta(header.meta, context);
        check_sizes(c.model, header.tensors, context);
        return c;
    }

    io::CheckpointFile file = io::read_checkpoint_file(path);
    Checkpoint c = checkpoint_from_meta(file.meta, context);

    // Index the file's tensors by name; the container guarantees unique names.
    std::unordered_map<std::string, Tensor*> by_name;
    for (io::NamedTensor& t : file.tensors) {
        by_name.emplace(t.name, &t.value);
    }
    const std::vector<nn::Parameter*> params = model->parameters();
    const auto take = [&](const std::string& name, const Shape& want) {
        const auto it = by_name.find(name);
        TITAN_CHECK(it != by_name.end(), context + ": missing tensor " + name);
        TITAN_CHECK(it->second->shape() == want,
                    shape_mismatch(context, name, it->second->shape(), want));
        Tensor* t = it->second;
        by_name.erase(it);
        return t;
    };

    // Check every tensor before touching the model, so a bad file leaves it
    // unchanged.
    std::vector<Tensor*> values, m, v;
    for (const nn::Parameter* p : params) {
        values.push_back(take(p->name, p->value.shape()));
    }
    for (const nn::Parameter* p : params) {
        m.push_back(take(std::string(kMomentM) + p->name, p->value.shape()));
    }
    for (const nn::Parameter* p : params) {
        v.push_back(take(std::string(kMomentV) + p->name, p->value.shape()));
    }
    if (!by_name.empty()) {
        // Name the first extra tensor in file order, for a stable message.
        for (const io::NamedTensor& t : file.tensors) {
            if (by_name.contains(t.name)) {
                throw_error(__FILE__, __LINE__,
                            context + ": unexpected tensor " + t.name +
                                " (not a parameter of the model or its "
                                "optimizer state)");
            }
        }
    }

    // Shapes match, but settings without a tensor of their own (rope_theta,
    // norm_eps, the norm or MLP kind of equal size) must match too.
    check_same_config(c.model, model->config(), context);

    // Replace each parameter's value in place: the Parameter objects stay, so
    // a tied head and any optimizer holding them see the new values.
    const Device device = model->device();
    for (std::size_t i = 0; i < params.size(); ++i) {
        params[i]->value = values[i]->to(device);
        c.state.optimizer.m.push_back(std::move(*m[i]));
        c.state.optimizer.v.push_back(std::move(*v[i]));
    }
    return c;
}

tokenizer::CharTokenizer checkpoint_tokenizer(const std::filesystem::path& path,
                                              const Checkpoint& ckpt) {
    const std::string context = file_context(path);
    std::optional<tokenizer::CharTokenizer> tok;
    try {
        tok = tokenizer::CharTokenizer::from_json(ckpt.tokenizer);
    } catch (const Error& e) {
        throw_error(__FILE__, __LINE__,
                    context + ": meta.tokenizer: " + e.what());
    }
    TITAN_CHECK(tok->vocab_size() == ckpt.model.vocab_size,
                context + ": the tokenizer has " +
                    std::to_string(tok->vocab_size()) +
                    " tokens but meta.vocab_size is " +
                    std::to_string(ckpt.model.vocab_size));
    return *std::move(tok);
}

}  // namespace titan::runtime
