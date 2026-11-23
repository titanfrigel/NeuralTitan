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
#include <optional>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "io/token_file.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "ops/matmul.hpp"
#include "runtime/checkpoint.hpp"
#include "runtime/generator.hpp"
#include "runtime/sampler.hpp"
#include "runtime/trainer.hpp"
#include "tokenizer/char_tokenizer.hpp"

namespace titan::cli {
namespace {

namespace fs = std::filesystem;

constexpr std::int64_t kSampleChars = 500;
constexpr float kSampleTemperature = 0.8f;
// titan generate defaults.
constexpr float kGenerateTemperature = 0.8f;
constexpr float kGenerateTopP = 0.9f;
constexpr std::int64_t kGenerateMaxNew = 500;

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
std::vector<std::int32_t> sample_prompt(const tokenizer::CharTokenizer& tok) {
    try {
        return tok.encode("\n");
    } catch (const Error&) {
        return {0};
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

// The tokenizer stored in a checkpoint, checked against the model's
// vocabulary size.
tokenizer::CharTokenizer checkpoint_tokenizer(const fs::path& path,
                                              const runtime::Checkpoint& ckpt) {
    auto tok = tokenizer::CharTokenizer::from_json(ckpt.tokenizer);
    TITAN_CHECK(tok.vocab_size() == ckpt.model.vocab_size,
                path.string() + ": the tokenizer has " +
                    std::to_string(tok.vocab_size()) +
                    " tokens but vocab_size is " +
                    std::to_string(ckpt.model.vocab_size));
    return tok;
}

// A token file of a run directory, with every id checked against the
// vocabulary (the model does not validate its inputs), and the stream checked
// against the length and CRC-32 the checkpoint saved for it.
std::vector<std::int32_t> read_run_tokens(const fs::path& path,
                                          std::int64_t vocab_size,
                                          std::int64_t saved_count,
                                          std::uint32_t saved_crc32) {
    std::vector<std::int32_t> ids = io::read_token_file(path);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        TITAN_CHECK(ids[i] >= 0 && ids[i] < vocab_size,
                    path.string() + ": token " + std::to_string(i) + " is " +
                        std::to_string(ids[i]) + ", outside [0, " +
                        std::to_string(vocab_size) + ")");
    }
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

namespace {

// Runs `trainer` (up to --stop-at when given), saving <out>/latest.ckpt at
// every checkpoint step. When the run reached its last step, it samples 500
// characters into <out>/sample.txt exactly as M1 did. Prints the summary.
void run_training(runtime::Trainer& trainer, model::Model& model,
                  const tokenizer::CharTokenizer& tok,
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
        // token 0. The sample keeps the M1 sampler and seed so that every
        // milestone's sample.txt stays reproducible.
        Rng sample_rng(tc.seed);
        const std::vector<std::int32_t> ids =
            runtime::generate_naive(model, sample_prompt(tok), kSampleChars,
                                    kSampleTemperature, sample_rng);
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

// `titan train --config ... --data ... --out ...`: a new run.
void train_new(const Flags& flags, Device device,
               std::optional<std::int64_t> stop_at) {
    std::optional<std::int64_t> steps;
    if (flags.contains("steps")) steps = parse_steps(flags.at("steps"));

    const fs::path out = flags.at("out");
    const io::JsonValue config =
        io::parse_json(io::read_file(flags.at("config")));
    const std::string corpus = io::read_file(flags.at("data"));

    // Everything that can fail on bad input runs before anything is written
    // or any training starts: config sections, --stop-at, tokenizer, corpus
    // split, model and trainer.

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
    runtime::TrainConfig tc = runtime::parse_train_config(train_section);
    tc.device = device;
    check_stop_at(stop_at, 0, tc.steps);

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

    // 4. Only now write the outputs, then train.
    fs::create_directories(out);
    if (!load_tokenizer) {
        io::write_file(out / "tokenizer.json", io::to_json(tok.to_json()));
    }
    io::write_token_file(out / "train.bin", train_tokens);
    io::write_token_file(out / "val.bin", val_tokens);
    run_training(trainer, model, tok, tc, out, stop_at);
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
    const auto tok = checkpoint_tokenizer(path, header);
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
        read_run_tokens(out / "train.bin", tok.vocab_size(),
                        header.state.train_tokens, header.state.train_crc32);
    const std::vector<std::int32_t> val_tokens =
        read_run_tokens(out / "val.bin", tok.vocab_size(),
                        header.state.val_tokens, header.state.val_crc32);

    model::Model model(header.model);
    runtime::Checkpoint ckpt = runtime::load_checkpoint(path, &model);
    runtime::Trainer trainer(model, tc, train_tokens, val_tokens, std::cout,
                             std::move(ckpt.state));
    std::cout << "resumed " << path.string() << " after step "
              << header.state.step << '\n';
    run_training(trainer, model, tok, tc, out, stop_at);
}

}  // namespace

void train(std::span<const std::string_view> args) {
    const Flags flags =
        parse_flags(args,
                    {"config", "data", "out", "tokenizer", "steps", "device",
                     "gemm", "resume", "stop-at"},
                    {});
    const bool resume = flags.contains("resume");
    if (resume) {
        // The run directory supplies everything these would set.
        for (const char* name :
             {"config", "data", "out", "tokenizer", "steps"}) {
            if (flags.contains(name)) throw UsageError{};
        }
    } else {
        for (const char* name : {"config", "data", "out"}) {
            if (!flags.contains(name)) throw UsageError{};
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

void generate(std::span<const std::string_view> args) {
    const Flags flags = parse_flags(args,
                                    {"checkpoint", "prompt", "max-new", "temp",
                                     "top-k", "top-p", "seed", "device"},
                                    {"checkpoint"}, {"no-cache"});

    // 1. Every flag, before any file is read.
    runtime::SamplerConfig sampler{kGenerateTemperature, 0, kGenerateTopP};
    if (flags.contains("temp")) {
        sampler.temperature = parse_float(flags.at("temp"));
    }
    if (flags.contains("top-k")) sampler.top_k = parse_steps(flags.at("top-k"));
    if (flags.contains("top-p")) sampler.top_p = parse_float(flags.at("top-p"));
    try {
        runtime::validate(sampler);
    } catch (const Error&) {
        throw UsageError{};
    }
    const std::int64_t max_new = flags.contains("max-new")
                                     ? parse_positive(flags.at("max-new"))
                                     : kGenerateMaxNew;
    std::optional<std::uint64_t> seed;
    if (flags.contains("seed")) seed = parse_seed(flags.at("seed"));
    const bool use_cache = !flags.contains("no-cache");
    const Device device = select_device(flags);

    // 2. The checkpoint header, the tokenizer and the prompt: a damaged file
    // or an unknown character fails here, before any output.
    const fs::path path = flags.at("checkpoint");
    const runtime::Checkpoint header = runtime::load_checkpoint(path, nullptr);
    const auto tok = checkpoint_tokenizer(path, header);
    std::vector<std::int32_t> prompt;
    if (flags.contains("prompt")) {
        try {
            prompt = tok.encode(flags.at("prompt"));
        } catch (const Error& e) {
            throw Error(std::string("--prompt: ") + e.what());
        }
        TITAN_CHECK(!prompt.empty(), "--prompt is empty");
    } else {
        prompt = sample_prompt(tok);
    }

    // 3. The model, moved to its device before the Generator builds its
    // cache there.
    model::Model model(header.model);
    runtime::load_checkpoint(path, &model);
    model.to(device);
    const std::uint64_t generation_seed = seed.value_or(header.train.seed);
    std::optional<runtime::Generator> generator;
    if (use_cache) generator.emplace(model, sampler, generation_seed);

    // 4. Echo the prompt, then stream each token as soon as it is sampled.
    std::cout << tok.decode(prompt) << std::flush;
    const auto on_token = [&](std::int32_t id) {
        std::cout << tok.decode(std::span(&id, 1)) << std::flush;
    };
    const auto start = std::chrono::steady_clock::now();
    if (generator) {
        generator->generate(prompt, max_new, on_token);
    } else {
        Rng rng(generation_seed);
        runtime::generate_naive(model, prompt, max_new, sampler, rng, on_token);
    }
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    std::cout << '\n' << std::flush;
    std::cerr << std::fixed << std::setprecision(1) << "tokens/s "
              << static_cast<double>(max_new) / elapsed.count() << " ("
              << max_new << " tokens in " << std::setprecision(3)
              << elapsed.count() << " s, " << (use_cache ? "cache" : "no cache")
              << ", " << to_string(device) << ")\n";
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
