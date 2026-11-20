#include "commands.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "io/token_file.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "runtime/sampler.hpp"
#include "runtime/trainer.hpp"
#include "tokenizer/char_tokenizer.hpp"

namespace titan::cli {
namespace {

namespace fs = std::filesystem;

constexpr std::int64_t kSampleChars = 500;
constexpr float kSampleTemperature = 0.8f;

using Flags = std::map<std::string, std::string, std::less<>>;

// Parses `--name value` pairs. Every flag must be in `allowed`, appear at most
// once and carry a value; each flag in `required` must be present.
Flags parse_flags(std::span<const std::string_view> args,
                  const std::vector<std::string>& allowed,
                  const std::vector<std::string>& required) {
    Flags flags;
    for (std::size_t i = 0; i < args.size(); i += 2) {
        const std::string name(args[i]);
        const bool known =
            name.starts_with("--") &&
            std::ranges::find(allowed, name.substr(2)) != allowed.end();
        if (!known || i + 1 >= args.size()) throw UsageError{};
        if (!flags.emplace(name.substr(2), std::string(args[i + 1])).second) {
            throw UsageError{};
        }
    }
    for (const std::string& name : required) {
        if (!flags.contains(name)) throw UsageError{};
    }
    return flags;
}

void make_parent_dir(const fs::path& file) {
    if (file.has_parent_path()) fs::create_directories(file.parent_path());
}

std::int64_t parse_steps(const std::string& text) {
    std::int64_t value = 0;
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    if (ec != std::errc{} || ptr != end) throw UsageError{};
    return value;
}

// The prompt the sample starts from: a newline when the vocabulary has one,
// otherwise token 0.
std::vector<std::int32_t> sample_prompt(const tokenizer::CharTokenizer& tok) {
    try {
        return tok.encode("\n");
    } catch (const Error&) {
        return {0};
    }
}

}  // namespace

void tokenizer_train(std::span<const std::string_view> args) {
    const Flags flags =
        parse_flags(args, {"type", "corpus", "out"}, {"type", "corpus", "out"});
    if (flags.at("type") != "char") throw UsageError{};

    const std::string corpus = io::read_file(flags.at("corpus"));
    const auto tok = tokenizer::CharTokenizer::build(corpus);
    const fs::path out = flags.at("out");
    make_parent_dir(out);
    io::write_file(out, io::to_json(tok.to_json()));
    std::cout << "vocab_size " << tok.vocab_size() << '\n';
}

void train(std::span<const std::string_view> args) {
    const Flags flags = parse_flags(
        args, {"config", "data", "out", "tokenizer", "steps", "device"},
        {"config", "data", "out"});
    const std::string device =
        flags.contains("device") ? flags.at("device") : "cpu";
    if (device != "cpu" && device != "cuda") throw UsageError{};
    std::optional<std::int64_t> steps;
    if (flags.contains("steps")) steps = parse_steps(flags.at("steps"));
    if (device == "cuda") throw Error("CUDA backend arrives in M2");

    const fs::path out = flags.at("out");
    const io::JsonValue config =
        io::parse_json(io::read_file(flags.at("config")));
    const std::string corpus = io::read_file(flags.at("data"));

    // Everything that can fail on bad input runs before anything is written
    // or any training starts: config sections, tokenizer, corpus split, model,
    // trainer and the sample prompt.

    // 1. The tokenizer: loaded, or built from the corpus (and saved below).
    const bool load_tokenizer = flags.contains("tokenizer");
    const auto tok = load_tokenizer
                         ? tokenizer::CharTokenizer::from_json(io::parse_json(
                               io::read_file(flags.at("tokenizer"))))
                         : tokenizer::CharTokenizer::build(corpus);

    // The train section, with --steps replacing train.steps before validation
    // so that warmup < steps is still enforced.
    io::JsonValue train_section = config.at("train");
    if (steps) {
        train_section.set("steps",
                          io::JsonValue::number(static_cast<double>(*steps)));
    }
    const runtime::TrainConfig tc = runtime::parse_train_config(train_section);

    // 2. Encode once; the last val_fraction of the tokens validates.
    std::vector<std::int32_t> tokens = tok.encode(corpus);
    const auto n = static_cast<std::int64_t>(tokens.size());
    const std::int64_t min_split = tc.seq + 2;
    const std::int64_t n_val = std::max(
        min_split, static_cast<std::int64_t>(
                       std::floor(static_cast<double>(n) * tc.val_fraction)));
    TITAN_CHECK(n - n_val >= min_split,
                "the corpus has " + std::to_string(n) +
                    " tokens, too few for a training split and a validation "
                    "split of at least seq + 2 = " +
                    std::to_string(min_split) + " tokens each");
    const std::vector<std::int32_t> train_tokens(tokens.begin(),
                                                 tokens.end() - n_val);
    const std::vector<std::int32_t> val_tokens(tokens.end() - n_val,
                                               tokens.end());

    // 3. The model and the trainer (which checks seq against context_length).
    model::Model model(
        model::parse_model_config(config.at("model"), tok.vocab_size()));
    Rng init_rng(tc.seed);
    model.init(init_rng);
    runtime::Trainer trainer(model, tc, train_tokens, val_tokens, std::cout);

    // 4. The sample prompt: a newline when the vocabulary has one, else token
    // 0.
    const std::vector<std::int32_t> prompt = sample_prompt(tok);

    // 5. Only now write the outputs, then train.
    fs::create_directories(out);
    if (!load_tokenizer) {
        io::write_file(out / "tokenizer.json", io::to_json(tok.to_json()));
    }
    io::write_token_file(out / "train.bin", train_tokens);
    io::write_token_file(out / "val.bin", val_tokens);
    const runtime::TrainResult result = trainer.run();

    // Sample 500 characters from the prompt.
    Rng sample_rng(tc.seed);
    const std::vector<std::int32_t> ids = runtime::generate_naive(
        model, prompt, kSampleChars, kSampleTemperature, sample_rng);
    const std::string sample = tok.decode(ids);
    io::write_file(out / "sample.txt", sample);
    std::cout << "--- sample ---\n" << sample << "\n--- end sample ---\n";

    // 6. Summary.
    std::cout << "params " << model.num_parameters() << '\n';
    std::cout << "time " << result.seconds << " s\n";
    std::cout << "final val_loss " << result.final_val_loss << '\n';
}

}  // namespace titan::cli
