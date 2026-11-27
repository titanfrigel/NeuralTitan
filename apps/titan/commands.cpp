#include "commands.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "core/utf8.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "io/token_file.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "ops/matmul.hpp"
#include "runtime/checkpoint.hpp"
#include "runtime/generator.hpp"
#include "runtime/hf_model.hpp"
#include "runtime/sampler.hpp"
#include "runtime/trainer.hpp"
#include "tokenizer/bpe_trainer.hpp"
#include "tokenizer/char_tokenizer.hpp"
#include "tokenizer/encode_chunked.hpp"
#include "tokenizer/tokenizer.hpp"

namespace titan::cli {
namespace {

namespace fs = std::filesystem;

// The largest corpus file the commands read. io::read_file's default cap is
// 1 GiB, below the 2.2 GB TinyStories train file; 4 GiB leaves room for it
// while still refusing a file that cannot be held in memory twice over.
// Config and tokenizer files keep the default cap.
constexpr std::size_t kMaxCorpusBytes = std::size_t{1} << 32;

constexpr std::int64_t kSampleTokens = 500;
constexpr float kSampleTemperature = 0.8f;
// titan generate defaults.
constexpr float kGenerateTemperature = 0.8f;
constexpr float kGenerateTopP = 0.9f;
constexpr std::int64_t kGenerateMaxNew = 500;
// titan tokenizer train --type bpe: <|endoftext|> and the 256 bytes, up to
// 2^18 tokens. The cap keeps any trained tokenizer inside the checkpoint's
// 16 MiB header: a token costs about 28 bytes of compact JSON (its vocabulary
// entry and its merge), so 262144 tokens need about 7 MiB.
constexpr std::int64_t kMinBpeVocab = 257;
constexpr std::int64_t kMaxBpeVocab = 262144;

using Flags = std::map<std::string, std::string, std::less<>>;

// Parses `--name value` pairs and bare `--name` switches. Every flag must be
// in `allowed` or `switches` and appear at most once; each flag in `allowed`
// carries a value (a switch is stored with an empty one), and each flag in
// `required` must be present.
Flags parse_flags(std::span<const std::string_view> args,
                  const std::vector<std::string>& allowed,
                  const std::vector<std::string>& required,
                  const std::vector<std::string>& switches = {}) {
    Flags flags;
    std::size_t i = 0;
    while (i < args.size()) {
        const std::string arg(args[i]);
        if (!arg.starts_with("--")) throw UsageError{};
        const std::string name = arg.substr(2);
        std::string value;
        if (std::ranges::find(switches, name) != switches.end()) {
            i += 1;
        } else if (std::ranges::find(allowed, name) != allowed.end() &&
                   i + 1 < args.size()) {
            value = std::string(args[i + 1]);
            i += 2;
        } else {
            throw UsageError{};
        }
        if (!flags.emplace(name, std::move(value)).second) throw UsageError{};
    }
    for (const std::string& name : required) {
        if (!flags.contains(name)) throw UsageError{};
    }
    return flags;
}

void make_parent_dir(const fs::path& file) {
    if (file.has_parent_path()) fs::create_directories(file.parent_path());
}

std::int64_t parse_steps(std::string_view text) {
    std::int64_t value = 0;
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    if (ec != std::errc{} || ptr != end) throw UsageError{};
    return value;
}

std::int64_t parse_positive(std::string_view text) {
    const std::int64_t value = parse_steps(text);
    if (value < 1) throw UsageError{};
    return value;
}

std::uint64_t parse_seed(std::string_view text) {
    std::uint64_t value = 0;
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    if (ec != std::errc{} || ptr != end) throw UsageError{};
    return value;
}

float parse_float(std::string_view text) {
    float value = 0.0f;
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    if (ec != std::errc{} || ptr != end) throw UsageError{};
    return value;
}

// "256,512,1024": a comma-separated list of positive integers.
std::vector<std::int64_t> parse_sizes(std::string_view text) {
    std::vector<std::int64_t> sizes;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = text.find(',', start);
        sizes.push_back(parse_positive(text.substr(start, comma - start)));
        if (comma == std::string_view::npos) return sizes;
        start = comma + 1;
    }
}

Device parse_device(std::string_view text) {
    if (text == "cpu") return Device::CPU;
    if (text == "cuda") return Device::CUDA;
    throw UsageError{};
}

ops::GemmBackend parse_gemm(std::string_view text) {
    if (text == "cublas") return ops::GemmBackend::cublas;
    if (text == "naive") return ops::GemmBackend::naive;
    if (text == "tiled") return ops::GemmBackend::tiled;
    throw UsageError{};
}

std::string_view gemm_name(ops::GemmBackend backend) {
    switch (backend) {
        case ops::GemmBackend::cublas:
            return "cublas";
        case ops::GemmBackend::naive:
            return "naive";
        case ops::GemmBackend::tiled:
            return "tiled";
    }
    return "unknown";
}

void require_gpu() {
    if (!cuda_available()) throw Error("no CUDA device available");
}

// An (n, n) matrix of N(0, 1) values on the GPU.
Tensor random_square(std::int64_t n, Rng& rng) {
    Tensor t = Tensor::empty({n, n}, Device::CPU);
    rng.fill_normal(t, 0.0f, 1.0f);
    return t.to(Device::CUDA);
}

// Milliseconds per out += a·b on the GPU with the current GEMM backend: one
// untimed warm-up call (cuBLAS picks its kernel, caches fill), then `repeat`
// calls timed on the host. Kernels run asynchronously on the default stream,
// so the timer starts and stops on cuda_synchronize().
double time_matmul_ms(Tensor& out, const Tensor& a, const Tensor& b,
                      std::int64_t repeat) {
    ops::matmul_acc(out, a, b, false, false);
    cuda_synchronize();
    const auto start = std::chrono::steady_clock::now();
    for (std::int64_t i = 0; i < repeat; ++i) {
        ops::matmul_acc(out, a, b, false, false);
    }
    cuda_synchronize();
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    return elapsed.count() / static_cast<double>(repeat);
}

// The prompt the sample starts from: a newline when the vocabulary has one,
// otherwise token 0.
std::vector<std::int32_t> sample_prompt(const tokenizer::Tokenizer& tok) {
    try {
        return tok.encode("\n");
    } catch (const Error&) {
        return {0};
    }
}

// The character or BPE tokenizer saved in `path`; errors name the file.
std::unique_ptr<tokenizer::Tokenizer> read_tokenizer(const std::string& path) {
    const std::string text = io::read_file(path);
    try {
        return tokenizer::load_tokenizer(io::parse_json(text));
    } catch (const Error& e) {
        throw Error(path + ": " + e.what());
    }
}

// The device from --device, or the GPU when there is one; checks that a
// requested GPU exists and applies --gemm. Runs before any file is read.
Device select_device(const Flags& flags) {
    const Device device = flags.contains("device")
                              ? parse_device(flags.at("device"))
                              : (cuda_available() ? Device::CUDA : Device::CPU);
    std::optional<ops::GemmBackend> gemm;
    if (flags.contains("gemm")) gemm = parse_gemm(flags.at("gemm"));
    if (device == Device::CUDA) require_gpu();
    // CPU matmuls ignore the GEMM backend, so only cuBLAS (the default, a
    // no-op) is accepted with the CPU.
    if (device == Device::CPU && gemm && *gemm != ops::GemmBackend::cublas) {
        throw Error("--gemm needs --device cuda");
    }
    if (gemm) ops::set_gemm_backend(*gemm);
    return device;
}

// A token file with every id checked against the vocabulary (the model does
// not validate its inputs).
std::vector<std::int32_t> read_checked_tokens(const fs::path& path,
                                              std::int64_t vocab_size) {
    std::vector<std::int32_t> ids = io::read_token_file(path);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        TITAN_CHECK(ids[i] >= 0 && ids[i] < vocab_size,
                    path.string() + ": token " + std::to_string(i) + " is " +
                        std::to_string(ids[i]) + ", outside [0, " +
                        std::to_string(vocab_size) + ")");
    }
    return ids;
}

// A token file of a run directory, with every id checked against the
// vocabulary, and the stream checked against the length and CRC-32 the
// checkpoint saved for it.
std::vector<std::int32_t> read_run_tokens(const fs::path& path,
                                          std::int64_t vocab_size,
                                          std::int64_t saved_count,
                                          std::uint32_t saved_crc32) {
    std::vector<std::int32_t> ids = read_checked_tokens(path, vocab_size);
    const auto count = static_cast<std::int64_t>(ids.size());
    const std::uint32_t crc = runtime::token_crc32(ids);
    TITAN_CHECK(count == saved_count && crc == saved_crc32,
                path.string() + ": the token file changed since the " +
                    "checkpoint (" + std::to_string(count) +
                    " tokens with crc32 " + std::to_string(crc) +
                    ", the checkpoint has " + std::to_string(saved_count) +
                    " tokens with crc32 " + std::to_string(saved_crc32) + ")");
    return ids;
}

// Checks --stop-at against the run: after the last completed step and no
// later than the last step.
void check_stop_at(std::optional<std::int64_t> stop_at, std::int64_t done,
                   std::int64_t steps) {
    if (!stop_at) return;
    TITAN_CHECK(*stop_at > done && *stop_at <= steps,
                "--stop-at " + std::to_string(*stop_at) +
                    " must be after the last completed step " +
                    std::to_string(done) + " and at most train.steps " +
                    std::to_string(steps));
}

}  // namespace

void tokenizer_train(std::span<const std::string_view> args) {
    const Flags flags = parse_flags(args, {"type", "corpus", "out", "vocab"},
                                    {"type", "corpus", "out"});
    const std::string& type = flags.at("type");
    // --vocab is required with bpe and meaningless with char.
    if (type != "char" && type != "bpe") throw UsageError{};
    if ((type == "bpe") != flags.contains("vocab")) throw UsageError{};
    std::int64_t vocab = 0;
    if (type == "bpe") {
        vocab = parse_steps(flags.at("vocab"));
        if (vocab < kMinBpeVocab || vocab > kMaxBpeVocab) throw UsageError{};
    }

    const std::string corpus =
        io::read_file(flags.at("corpus"), kMaxCorpusBytes);
    const fs::path out = flags.at("out");
    if (type == "char") {
        const auto tok = tokenizer::CharTokenizer::build(corpus);
        make_parent_dir(out);
        io::write_file(out, io::to_json(tok.to_json()));
        std::cout << "vocab_size " << tok.vocab_size() << '\n';
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto tok = tokenizer::train_bpe(corpus, vocab);
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    make_parent_dir(out);
    io::write_file(out, io::to_json(tok.to_json()));
    std::cout << "vocab_size " << tok.vocab_size() << '\n'
              << "merges " << tok.num_merges() << '\n'
              << "time " << std::fixed << std::setprecision(2)
              << elapsed.count() << " s\n";
}

void tokenizer_encode(std::span<const std::string_view> args) {
    const Flags flags = parse_flags(args, {"tokenizer", "text", "file", "out"},
                                    {"tokenizer"}, {"summary"});
    // Exactly one source of text.
    if (flags.contains("text") == flags.contains("file")) throw UsageError{};

    const std::unique_ptr<tokenizer::Tokenizer> tok =
        read_tokenizer(flags.at("tokenizer"));
    const std::string text =
        flags.contains("text")
            ? flags.at("text")
            : io::read_file(flags.at("file"), kMaxCorpusBytes);
    const std::vector<std::int32_t> ids = tokenizer::encode_chunked(*tok, text);

    // With --out the ids go to a token file and only the summary is printed.
    const bool to_file = flags.contains("out");
    if (to_file) {
        const fs::path out = flags.at("out");
        make_parent_dir(out);
        io::write_token_file(out, ids);
    }
    if (!to_file && !flags.contains("summary")) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) std::cout << ' ';
            std::cout << ids[i];
        }
        std::cout << '\n';
        return;
    }
    // Bytes per token: the compression the tokenizer achieves on this text
    // (0 when there is no token).
    const double bytes_per_token = ids.empty()
                                       ? 0.0
                                       : static_cast<double>(text.size()) /
                                             static_cast<double>(ids.size());
    std::cout << "tokens " << ids.size() << '\n'
              << "crc32 0x" << std::hex << std::setw(8) << std::setfill('0')
              << runtime::token_crc32(ids) << std::dec << '\n'
              << "bytes_per_token " << std::fixed << std::setprecision(3)
              << bytes_per_token << '\n';
}

namespace {

// Runs `trainer` (up to --stop-at when given), saving <out>/latest.ckpt at
// every checkpoint step. When the run reached its last step, it samples 500
// tokens (characters with a character tokenizer) into <out>/sample.txt
// exactly as M1 did. Prints the summary.
void run_training(runtime::Trainer& trainer, model::Model& model,
                  const tokenizer::Tokenizer& tok,
                  const runtime::TrainConfig& tc, const fs::path& out,
                  std::optional<std::int64_t> stop_at) {
    const io::JsonValue tok_json = tok.to_json();
    trainer.on_checkpoint([&](std::int64_t) {
        runtime::save_checkpoint(
            out / "latest.ckpt", model,
            runtime::Checkpoint{model.config(), tc, tok_json, trainer.state()});
    });
    const runtime::TrainResult result = trainer.run(stop_at);
    const bool finished = trainer.state().step == tc.steps;

    if (finished) {
        // The sample prompt: a newline when the vocabulary has one, else
        // token 0. The sample keeps the M1 sampler (temperature only: top-k
        // off, top-p 1) and seed so that every milestone's sample.txt stays
        // reproducible. A BPE sample ends early at end of text.
        Rng sample_rng(tc.seed);
        const std::vector<std::int32_t> ids = runtime::generate_naive(
            model, sample_prompt(tok), kSampleTokens,
            runtime::SamplerConfig{kSampleTemperature, 0, 1.0f}, sample_rng, {},
            tok.eos_id());
        const std::string sample = tok.decode(ids);
        io::write_file(out / "sample.txt", sample);
        std::cout << "--- sample ---\n" << sample << "\n--- end sample ---\n";
    } else {
        std::cout << "stopped after step " << trainer.state().step
                  << "; continue with titan train --resume " << out.string()
                  << '\n';
    }

    std::cout << "device " << to_string(tc.device);
    if (tc.device == Device::CUDA) {
        std::cout << ", gemm " << gemm_name(ops::gemm_backend());
    }
    std::cout << '\n';
    std::cout << "params " << model.num_parameters() << '\n';
    std::cout << "time " << result.seconds << " s\n";
    if (finished) {
        std::cout << "final val_loss " << result.final_val_loss << '\n';
    }
}

// `titan train --config ... (--data ... | --train-tokens ... --val-tokens ...)
// --out ...`: a new run.
void train_new(const Flags& flags, Device device,
               std::optional<std::int64_t> stop_at) {
    std::optional<std::int64_t> steps;
    if (flags.contains("steps")) steps = parse_steps(flags.at("steps"));

    const fs::path out = flags.at("out");
    const bool from_tokens = flags.contains("train-tokens");
    const io::JsonValue config =
        io::parse_json(io::read_file(flags.at("config")));
    const std::string corpus =
        from_tokens ? std::string()
                    : io::read_file(flags.at("data"), kMaxCorpusBytes);

    // Everything that can fail on bad input runs before anything is written
    // or any training starts: config sections, --stop-at, tokenizer, token
    // split, model and trainer.

    // 1. The tokenizer: loaded, or built from the corpus (and saved below).
    const bool load_tokenizer = flags.contains("tokenizer");
    const std::unique_ptr<tokenizer::Tokenizer> tok =
        load_tokenizer ? read_tokenizer(flags.at("tokenizer"))
                       : std::make_unique<tokenizer::CharTokenizer>(
                             tokenizer::CharTokenizer::build(corpus));

    // The train section, with --steps replacing train.steps before validation
    // so that warmup < steps is still enforced.
    io::JsonValue train_section = config.at("train");
    if (steps) {
        train_section.set("steps",
                          io::JsonValue::number(static_cast<double>(*steps)));
    }
    runtime::TrainConfig tc = runtime::parse_train_config(train_section);
    tc.device = device;
    check_stop_at(stop_at, 0, tc.steps);

    // 2. The two token streams. Token files are used as they are (val_fraction
    // does not apply), each checked to hold at least seq + 2 tokens. A corpus
    // is encoded once; its last val_fraction of the tokens validates.
    const std::int64_t min_split = tc.seq + 2;
    std::vector<std::int32_t> train_tokens;
    std::vector<std::int32_t> val_tokens;
    if (from_tokens) {
        const std::int64_t vocab = tok->vocab_size();
        const std::pair<const char*, std::vector<std::int32_t>*> splits[] = {
            {"train", &train_tokens}, {"val", &val_tokens}};
        for (const auto& [name, ids] : splits) {
            const std::string flag = std::string(name) + "-tokens";
            *ids = read_checked_tokens(flags.at(flag), vocab);
            TITAN_CHECK(static_cast<std::int64_t>(ids->size()) >= min_split,
                        "--" + flag + " " + flags.at(flag) + " has " +
                            std::to_string(ids->size()) +
                            " tokens, too few for a " + name +
                            " split of at least seq + 2 = " +
                            std::to_string(min_split) + " tokens");
        }
    } else {
        const std::vector<std::int32_t> tokens =
            tokenizer::encode_chunked(*tok, corpus);
        const auto n = static_cast<std::int64_t>(tokens.size());
        const std::int64_t n_val =
            std::max(min_split, static_cast<std::int64_t>(std::floor(
                                    static_cast<double>(n) * tc.val_fraction)));
        TITAN_CHECK(n - n_val >= min_split,
                    "the corpus has " + std::to_string(n) +
                        " tokens, too few for a training split and a "
                        "validation split of at least seq + 2 = " +
                        std::to_string(min_split) + " tokens each");
        train_tokens.assign(tokens.begin(), tokens.end() - n_val);
        val_tokens.assign(tokens.end() - n_val, tokens.end());
    }

    // 3. The model and the trainer (which checks seq against context_length).
    model::Model model(
        model::parse_model_config(config.at("model"), tok->vocab_size()));
    Rng init_rng(tc.seed);
    model.init(init_rng);
    // Token files are moved into the trainer; the --data split is written to
    // the run directory below, so the trainer gets copies of it.
    runtime::Trainer trainer(
        model, tc, from_tokens ? std::move(train_tokens) : train_tokens,
        from_tokens ? std::move(val_tokens) : val_tokens, std::cout);

    // 4. Only now write the outputs, then train. --resume reads train.bin and
    // val.bin, so token files are copied into the run directory unchanged.
    fs::create_directories(out);
    if (!load_tokenizer) {
        io::write_file(out / "tokenizer.json", io::to_json(tok->to_json()));
    }
    if (from_tokens) {
        fs::copy_file(flags.at("train-tokens"), out / "train.bin",
                      fs::copy_options::overwrite_existing);
        fs::copy_file(flags.at("val-tokens"), out / "val.bin",
                      fs::copy_options::overwrite_existing);
    } else {
        io::write_token_file(out / "train.bin", train_tokens);
        io::write_token_file(out / "val.bin", val_tokens);
    }
    run_training(trainer, model, *tok, tc, out, stop_at);
}

// `titan train --resume <dir>`: continues the run saved in <dir>/latest.ckpt
// on <dir>/train.bin and <dir>/val.bin. The configs and the tokenizer come
// from the checkpoint; only the device is chosen anew.
void train_resume(const Flags& flags, Device device,
                  std::optional<std::int64_t> stop_at) {
    const fs::path out = flags.at("resume");
    const fs::path path = out / "latest.ckpt";

    // The header first: it gives the model config to build.
    const runtime::Checkpoint header = runtime::load_checkpoint(path, nullptr);
    const std::unique_ptr<tokenizer::Tokenizer> tok =
        runtime::checkpoint_tokenizer(path, header);
    runtime::TrainConfig tc = header.train;
    tc.device = device;
    check_stop_at(stop_at, header.state.step, tc.steps);
    // A finished run has no step left: resuming it would only rewrite the
    // sample and report a validation loss that was never computed.
    TITAN_CHECK(header.state.step < tc.steps,
                path.string() + ": the run already finished at step " +
                    std::to_string(header.state.step) + " of " +
                    std::to_string(tc.steps));
    const std::vector<std::int32_t> train_tokens =
        read_run_tokens(out / "train.bin", tok->vocab_size(),
                        header.state.train_tokens, header.state.train_crc32);
    const std::vector<std::int32_t> val_tokens =
        read_run_tokens(out / "val.bin", tok->vocab_size(),
                        header.state.val_tokens, header.state.val_crc32);

    model::Model model(header.model);
    runtime::Checkpoint ckpt = runtime::load_checkpoint(path, &model);
    runtime::Trainer trainer(model, tc, train_tokens, val_tokens, std::cout,
                             std::move(ckpt.state));
    std::cout << "resumed " << path.string() << " after step "
              << header.state.step << '\n';
    run_training(trainer, model, *tok, tc, out, stop_at);
}

}  // namespace

void train(std::span<const std::string_view> args) {
    const Flags flags =
        parse_flags(args,
                    {"config", "data", "out", "tokenizer", "steps", "device",
                     "gemm", "resume", "stop-at", "train-tokens", "val-tokens"},
                    {});
    const bool resume = flags.contains("resume");
    if (resume) {
        // The run directory supplies everything these would set.
        for (const char* name : {"config", "data", "out", "tokenizer", "steps",
                                 "train-tokens", "val-tokens"}) {
            if (flags.contains(name)) throw UsageError{};
        }
    } else {
        for (const char* name : {"config", "out"}) {
            if (!flags.contains(name)) throw UsageError{};
        }
        // Exactly one source: --data, or both token files (which need the
        // tokenizer they were encoded with).
        const bool tokens =
            flags.contains("train-tokens") || flags.contains("val-tokens");
        if (tokens) {
            if (flags.contains("data") || !flags.contains("train-tokens") ||
                !flags.contains("val-tokens") || !flags.contains("tokenizer")) {
                throw UsageError{};
            }
        } else if (!flags.contains("data")) {
            throw UsageError{};
        }
    }
    std::optional<std::int64_t> stop_at;
    if (flags.contains("stop-at")) {
        stop_at = parse_positive(flags.at("stop-at"));
    }
    const Device device = select_device(flags);
    if (resume) {
        train_resume(flags, device, stop_at);
    } else {
        train_new(flags, device, stop_at);
    }
}

namespace {

// The prompt ids: --prompt encoded, or sample_prompt without it. An unknown
// character or an empty prompt fails here, before any output.
std::vector<std::int32_t> generation_prompt(const Flags& flags,
                                            const tokenizer::Tokenizer& tok) {
    if (!flags.contains("prompt")) return sample_prompt(tok);
    std::vector<std::int32_t> prompt;
    try {
        prompt = tok.encode(flags.at("prompt"));
    } catch (const Error& e) {
        throw Error(std::string("--prompt: ") + e.what());
    }
    TITAN_CHECK(!prompt.empty(), "--prompt is empty");
    return prompt;
}

// What titan generate's flags ask for, independent of where the model comes
// from.
struct GenerateOptions {
    runtime::SamplerConfig sampler;
    std::int64_t max_new{};
    std::optional<std::uint64_t> seed;
    bool use_cache{};
    Device device{};
};

// Moves `model` to options.device, echoes the prompt, then streams each
// token as soon as it is sampled and writes tokens/s to stderr. `seed` is the
// generation seed (options.seed has already been applied by the caller).
void stream_generation(model::Model& model, const tokenizer::Tokenizer& tok,
                       const std::vector<std::int32_t>& prompt,
                       const GenerateOptions& options, std::uint64_t seed) {
    // The model is moved to its device before the Generator builds its cache
    // there.
    model.to(options.device);
    std::optional<runtime::Generator> generator;
    if (options.use_cache) generator.emplace(model, options.sampler, seed);

    // A BPE token may hold only part of a UTF-8 character, so the decoded
    // bytes go through a Utf8Stream: each step prints the complete part and
    // the stream keeps the incomplete tail until a later token completes it.
    Utf8Stream stream;
    std::cout << stream.push(tok.decode(prompt)) << std::flush;
    std::int64_t generated = 0;
    const auto on_token = [&](std::int32_t id) {
        ++generated;
        std::cout << stream.push(tok.decode(std::span(&id, 1))) << std::flush;
    };
    // The end-of-text token, when the tokenizer has one, ends generation.
    const std::optional<std::int32_t> stop_id = tok.eos_id();
    const auto start = std::chrono::steady_clock::now();
    if (generator) {
        generator->generate(prompt, options.max_new, on_token, stop_id);
    } else {
        Rng rng(seed);
        runtime::generate_naive(model, prompt, options.max_new, options.sampler,
                                rng, on_token, stop_id);
    }
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    // Whatever is still pending is a character the model never finished;
    // its bytes are printed as they are.
    std::cout << stream.finish() << '\n' << std::flush;
    std::cerr << std::fixed << std::setprecision(1) << "tokens/s "
              << static_cast<double>(generated) / elapsed.count() << " ("
              << generated << " tokens in " << std::setprecision(3)
              << elapsed.count() << " s, "
              << (options.use_cache ? "cache" : "no cache") << ", "
              << to_string(options.device) << ")\n";
}

// titan generate --checkpoint: the header, the tokenizer and the prompt are
// read before the weights, so a damaged file or an unknown character fails
// before any output.
void generate_from_checkpoint(const fs::path& path, const Flags& flags,
                              const GenerateOptions& options) {
    const runtime::Checkpoint header = runtime::load_checkpoint(path, nullptr);
    const std::unique_ptr<tokenizer::Tokenizer> tok =
        runtime::checkpoint_tokenizer(path, header);
    const std::vector<std::int32_t> prompt = generation_prompt(flags, *tok);
    model::Model model(header.model);
    runtime::load_checkpoint(path, &model);
    stream_generation(model, *tok, prompt, options,
                      options.seed.value_or(header.train.seed));
}

// titan generate --hf-model: load_hf_model reads and validates every file
// (config.json first) on the CPU before the prompt is encoded; any failure
// comes before output. A Hugging Face model has no training seed, so the
// seed defaults to 0.
void generate_from_hf_model(const fs::path& dir, const Flags& flags,
                            const GenerateOptions& options) {
    runtime::HfModel hf = runtime::load_hf_model(dir);
    const std::vector<std::int32_t> prompt =
        generation_prompt(flags, *hf.tokenizer);
    stream_generation(*hf.model, *hf.tokenizer, prompt, options,
                      options.seed.value_or(0));
}

}  // namespace

void generate(std::span<const std::string_view> args) {
    const Flags flags =
        parse_flags(args,
                    {"checkpoint", "hf-model", "prompt", "max-new", "temp",
                     "top-k", "top-p", "seed", "device"},
                    {}, {"no-cache"});
    // Exactly one model source.
    if (flags.contains("checkpoint") == flags.contains("hf-model")) {
        throw UsageError{};
    }

    // 1. Every flag, before any file is read.
    GenerateOptions options;
    options.sampler = {kGenerateTemperature, 0, kGenerateTopP};
    if (flags.contains("temp")) {
        options.sampler.temperature = parse_float(flags.at("temp"));
    }
    if (flags.contains("top-k")) {
        options.sampler.top_k = parse_steps(flags.at("top-k"));
    }
    if (flags.contains("top-p")) {
        options.sampler.top_p = parse_float(flags.at("top-p"));
    }
    try {
        runtime::validate(options.sampler);
    } catch (const Error&) {
        throw UsageError{};
    }
    options.max_new = flags.contains("max-new")
                          ? parse_positive(flags.at("max-new"))
                          : kGenerateMaxNew;
    if (flags.contains("seed")) options.seed = parse_seed(flags.at("seed"));
    options.use_cache = !flags.contains("no-cache");
    options.device = select_device(flags);

    // 2. The model, its tokenizer and the prompt, then the stream.
    if (flags.contains("checkpoint")) {
        generate_from_checkpoint(flags.at("checkpoint"), flags, options);
    } else {
        generate_from_hf_model(flags.at("hf-model"), flags, options);
    }
}

void bench(std::span<const std::string_view> args) {
    if (args.empty() || args[0] != "matmul") throw UsageError{};
    const Flags flags = parse_flags(args.subspan(1), {"sizes", "repeat"}, {});
    const std::vector<std::int64_t> sizes =
        flags.contains("sizes")
            ? parse_sizes(flags.at("sizes"))
            : std::vector<std::int64_t>{256, 512, 1024, 2048, 4096};
    const std::int64_t repeat =
        flags.contains("repeat") ? parse_positive(flags.at("repeat")) : 10;
    require_gpu();
    const CudaDeviceInfo info = *cuda_device_info();

    // Peak FP32 throughput: every SM of an sm_86 GPU has 128 FP32 lanes, and
    // each lane completes one fused multiply-add (2 FLOP) per clock, so
    // peak = SMs × 128 × 2 × clock. The 128 lanes are an sm_86 figure; other
    // architectures differ.
    constexpr double kFp32LanesPerSm = 128.0;
    const double clock_hz = info.clock_khz * 1e3;
    const double peak_gflops = static_cast<double>(info.sm_count) *
                               kFp32LanesPerSm * 2.0 * clock_hz / 1e9;
    // Peak memory bandwidth: GDDR memory transfers on both clock edges, each
    // transfer moving bus_width / 8 bytes, so
    // peak = 2 × memory clock × bus width / 8.
    const double peak_gbs = 2.0 * info.memory_clock_khz * 1e3 *
                            (info.memory_bus_width_bits / 8.0) / 1e9;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "device " << info.name << ", compute " << info.compute_major
              << '.' << info.compute_minor << ", " << info.sm_count
              << " SMs at " << info.clock_khz / 1000 << " MHz\n";
    std::cout << "peak FP32 " << peak_gflops << " GFLOP/s, peak bandwidth "
              << peak_gbs << " GB/s, ridge point " << peak_gflops / peak_gbs
              << " FLOP/byte\n";
    std::cout << "size backend ms GFLOP/s %peak\n";

    const ops::GemmBackend previous = ops::gemm_backend();
    Rng rng(1);
    for (const std::int64_t n : sizes) {
        const Tensor a = random_square(n, rng);
        const Tensor b = random_square(n, rng);
        Tensor out = Tensor::zeros({n, n}, Device::CUDA);
        // A square GEMM is n³ multiply-adds: 2n³ FLOP.
        const double flop = 2.0 * static_cast<double>(n) *
                            static_cast<double>(n) * static_cast<double>(n);
        for (const ops::GemmBackend backend :
             {ops::GemmBackend::cublas, ops::GemmBackend::naive,
              ops::GemmBackend::tiled}) {
            ops::set_gemm_backend(backend);
            const double ms = time_matmul_ms(out, a, b, repeat);
            const double gflops = flop / (ms * 1e-3) / 1e9;
            std::cout << n << ' ' << gemm_name(backend) << ' '
                      << std::setprecision(3) << ms << ' '
                      << std::setprecision(1) << gflops << ' '
                      << 100.0 * gflops / peak_gflops << '\n';
        }
    }
    ops::set_gemm_backend(previous);
}

}  // namespace titan::cli
