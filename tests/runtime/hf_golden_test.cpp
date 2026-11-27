// Parity with Hugging Face: the logits and greedy generations of the tiny
// Llama fixtures and of SmolLM2-135M, against the golden fixtures written by
// tools/make_llama_fixtures.py, on the CPU and on CUDA. The tolerance and the
// near-tie rule are those of docs/math/hf-llama.md, "Tolerance".

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "io/safetensors.hpp"
#include "model/model.hpp"
#include "nn/module.hpp"
#include "runtime/generator.hpp"
#include "runtime/hf_model.hpp"
#include "runtime/sampler.hpp"
#include "support/golden.hpp"
#include "tokenizer/tokenizer.hpp"

namespace {

using titan::Device;
using titan::io::JsonValue;
using titan::io::SafetensorsFile;
using titan::runtime::HfModel;
using titan::test::compare_greedy;
using titan::test::compare_logits;
using titan::test::GreedyCheck;
using titan::test::LogitCheck;

const std::filesystem::path kFixtures = TITAN_FIXTURES_DIR;
const std::filesystem::path kSmolLM2Dir =
    std::filesystem::path(TITAN_SOURCE_DIR) / "models" / "SmolLM2-135M";

// ---------------------------------------------------------------------------
// The helpers.

TITAN_TEST(Golden, CompareLogits) {
    // Below 1 the scale is 1, so the bound is the tolerance itself.
    const std::vector<float> small_ref = {0.1f, -0.5f};
    LogitCheck c =
        compare_logits(std::vector<float>{0.1f, -0.5009f}, small_ref, 1e-3);
    TITAN_EXPECT_EQ(c.scale, 1.0);
    TITAN_EXPECT(c.ok);
    c = compare_logits(std::vector<float>{0.1f, -0.5011f}, small_ref, 1e-3);
    TITAN_EXPECT(!c.ok);

    // The scale is the largest |ref|: 20 here, so the bound is 0.02 and an
    // error of 0.015 passes where an absolute 1e-3 would fail.
    const std::vector<float> ref = {10.0f, -20.0f, 3.0f};
    c = compare_logits(std::vector<float>{10.015f, -20.0f, 3.0f}, ref, 1e-3);
    TITAN_EXPECT_EQ(c.scale, 20.0);
    TITAN_EXPECT_NEAR(c.max_abs_diff, 0.015, 1e-6);
    TITAN_EXPECT(c.ok);
    c = compare_logits(std::vector<float>{10.0f, -20.0f, 3.025f}, ref, 1e-3);
    TITAN_EXPECT_NEAR(c.max_abs_diff, 0.025, 1e-6);
    TITAN_EXPECT(!c.ok);

    // Identical values pass with no error.
    c = compare_logits(ref, ref, 1e-3);
    TITAN_EXPECT_EQ(c.max_abs_diff, 0.0);
    TITAN_EXPECT(c.ok);

    // A NaN on either side fails.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    TITAN_EXPECT(
        !compare_logits(std::vector<float>{10.0f, nan, 3.0f}, ref, 1e-3).ok);
    TITAN_EXPECT(
        !compare_logits(ref, std::vector<float>{10.0f, nan, 3.0f}, 1e-3).ok);

    TITAN_EXPECT_THROWS(compare_logits(std::vector<float>{1.0f}, ref, 1e-3),
                        titan::Error);
}

TITAN_TEST(Golden, CompareGreedy) {
    const std::vector<std::int32_t> ref = {5, 9, 2, 7};
    const std::vector<double> gaps = {0.5, 0.0005, 0.0, 0.5};

    // Every step matches.
    GreedyCheck g = compare_greedy(ref, ref, gaps, 1e-3);
    TITAN_EXPECT_EQ(g.matched, std::int64_t{4});
    TITAN_EXPECT(!g.diverged_at_near_tie);
    TITAN_EXPECT(g.ok);

    // A mismatch at step 1, whose gap 5e-4 is below the tolerance: accepted,
    // and the comparison stops there.
    g = compare_greedy(std::vector<std::int32_t>{5, 8, 0, 0}, ref, gaps, 1e-3);
    TITAN_EXPECT_EQ(g.matched, std::int64_t{1});
    TITAN_EXPECT(g.diverged_at_near_tie);
    TITAN_EXPECT(g.ok);

    // An exact tie (gap 0) at step 2.
    g = compare_greedy(std::vector<std::int32_t>{5, 9, 3, 7}, ref, gaps, 1e-3);
    TITAN_EXPECT_EQ(g.matched, std::int64_t{2});
    TITAN_EXPECT(g.diverged_at_near_tie);
    TITAN_EXPECT(g.ok);

    // A mismatch at step 3, whose gap 0.5 is no near-tie.
    g = compare_greedy(std::vector<std::int32_t>{5, 9, 2, 6}, ref, gaps, 1e-3);
    TITAN_EXPECT_EQ(g.matched, std::int64_t{3});
    TITAN_EXPECT(!g.diverged_at_near_tie);
    TITAN_EXPECT(!g.ok);

    // "Below" is strict: a gap equal to the tolerance is no near-tie.
    g = compare_greedy(std::vector<std::int32_t>{6},
                       std::vector<std::int32_t>{5}, std::vector<double>{1e-3},
                       1e-3);
    TITAN_EXPECT(!g.ok);

    // Our run ending early is a mismatch at the first missing step.
    g = compare_greedy(std::vector<std::int32_t>{5}, ref, gaps, 1e-3);
    TITAN_EXPECT_EQ(g.matched, std::int64_t{1});
    TITAN_EXPECT(g.ok);  // step 1 is a near-tie
    g = compare_greedy(std::vector<std::int32_t>{5, 9, 2}, ref, gaps, 1e-3);
    TITAN_EXPECT_EQ(g.matched, std::int64_t{3});
    TITAN_EXPECT(!g.ok);

    TITAN_EXPECT_THROWS(
        compare_greedy(ref, ref, std::vector<double>{0.5}, 1e-3), titan::Error);
}

// ---------------------------------------------------------------------------
// The fixtures.

struct GoldenFiles {
    JsonValue json;
    SafetensorsFile tensors;
};

GoldenFiles load_golden(const std::filesystem::path& dir) {
    return {titan::io::parse_json(titan::io::read_file(dir / "golden.json")),
            SafetensorsFile::open(dir / "golden.safetensors")};
}

// The tolerance of the pass conditions (docs/math/hf-llama.md). The fixture
// records the tolerance it was made for; a different value there fails here
// instead of silently changing the bound.
constexpr double kTolerance = 1e-3;

double tolerance_of(const GoldenFiles& golden) {
    TITAN_EXPECT_EQ(golden.json.at("tolerance").as_number(), kTolerance);
    return kTolerance;
}

std::vector<float> read_tensor(const SafetensorsFile& file,
                               const std::string& name) {
    const auto* entry = file.find(name);
    TITAN_CHECK(entry != nullptr, "golden.safetensors: no tensor " + name);
    return file.read_f32(*entry);
}

std::vector<std::int32_t> int_list(const JsonValue& array) {
    std::vector<std::int32_t> out;
    for (const JsonValue& v : array.as_array()) {
        out.push_back(static_cast<std::int32_t>(v.as_int()));
    }
    return out;
}

std::vector<double> number_list(const JsonValue& array) {
    std::vector<double> out;
    for (const JsonValue& v : array.as_array()) out.push_back(v.as_number());
    return out;
}

// The model loaded from `dir` and moved to `device`.
HfModel load_on(const std::filesystem::path& dir, Device device) {
    HfModel hf = titan::runtime::load_hf_model(dir);
    hf.model->to(device);
    return hf;
}

// The logits (T, vocab) of one inference forward over the whole prompt,
// copied to the host.
std::vector<float> prompt_logits(titan::model::Model& model,
                                 const std::vector<std::int32_t>& ids) {
    titan::nn::ForwardContext ctx;
    ctx.training = false;
    const auto t = static_cast<std::int64_t>(ids.size());
    return model.forward(ids, 1, t, ctx).to_vector();
}

void print_logits(const std::string& label, const LogitCheck& c) {
    std::cout << "  " << label << ": max_abs_diff / scale "
              << c.max_abs_diff / c.scale << " (" << c.max_abs_diff << " / "
              << c.scale << ")\n";
}

// Runs our cached greedy generation for one fixture prompt and compares it
// with the reference. Both sides list one token per generated step, ending
// with eos when the run stopped there: the reference has one gap more than
// tokens then, and our Generator returns fewer than the requested tokens.
GreedyCheck check_greedy(HfModel& hf, const JsonValue& prompt, double tol,
                         const std::string& label) {
    const std::vector<std::int32_t> ids = int_list(prompt.at("ids"));
    std::vector<std::int32_t> ref = int_list(prompt.at("greedy"));
    const std::vector<double> gaps = number_list(prompt.at("greedy_gaps"));
    const std::int32_t eos = hf.config.eos_id;
    if (gaps.size() == ref.size() + 1) ref.push_back(eos);

    const auto steps = static_cast<std::int64_t>(gaps.size());
    titan::runtime::Generator gen(*hf.model, {0.0f, 0, 1.0f}, 0);
    const std::vector<std::int32_t> out = gen.generate(ids, steps, {}, eos);
    std::vector<std::int32_t> ours(out.begin() + ids.size(), out.end());
    if (static_cast<std::int64_t>(ours.size()) < steps) ours.push_back(eos);

    const GreedyCheck g = compare_greedy(ours, ref, gaps, tol);
    std::cout << "  " << label << ": greedy " << g.matched << " / "
              << ref.size();
    if (g.matched < static_cast<std::int64_t>(ref.size())) {
        std::cout << (g.diverged_at_near_tie ? ", near-tie" : ", diverged")
                  << " at step " << g.matched << " (gap " << gaps[g.matched]
                  << ")";
    }
    std::cout << "\n";
    return g;
}

const char* device_name(Device device) {
    return device == Device::CPU ? "cpu" : "cuda";
}

// ---------------------------------------------------------------------------
// The prompts: every fixture prompt's ids are what our tokenizer makes of its
// text, so the parity tests feed the model the reference's input. SmolLM2's
// tokenizer is the committed copy, so this runs without the weights.

void expect_prompt_ids(const std::filesystem::path& tokenizer_path,
                       const JsonValue& prompts) {
    const auto tok = titan::tokenizer::load_tokenizer(
        titan::io::parse_json(titan::io::read_file(tokenizer_path)));
    for (const JsonValue& prompt : prompts.as_array()) {
        TITAN_EXPECT(tok->encode(prompt.at("text").as_string()) ==
                     int_list(prompt.at("ids")));
    }
}

TITAN_TEST(HfGolden, PromptIdsMatchTokenizer) {
    const GoldenFiles tiny = load_golden(kFixtures / "tiny_llama");
    tolerance_of(tiny);
    for (const char* name : {"tied", "untied"}) {
        expect_prompt_ids(kFixtures / "tiny_llama" / name / "tokenizer.json",
                          tiny.json.at(name));
    }
    const GoldenFiles smollm2 = load_golden(kFixtures / "smollm2");
    tolerance_of(smollm2);
    expect_prompt_ids(kFixtures / "smollm2" / "tokenizer.json",
                      smollm2.json.at("prompts"));
}

// ---------------------------------------------------------------------------
// The tiny models: full logits at every prompt position, 16 greedy tokens.

void tiny_logits(Device device) {
    const std::filesystem::path dir = kFixtures / "tiny_llama";
    const GoldenFiles golden = load_golden(dir);
    const double tol = tolerance_of(golden);
    for (const char* name : {"tied", "untied"}) {
        HfModel hf = load_on(dir / name, device);
        const auto& prompts = golden.json.at(name).as_array();
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const std::string p = std::string(name) + ".p" + std::to_string(i);
            const std::vector<float> ours =
                prompt_logits(*hf.model, int_list(prompts[i].at("ids")));
            const LogitCheck c = compare_logits(
                ours, read_tensor(golden.tensors, p + ".logits"), tol);
            print_logits(std::string("tiny ") + p + " " + device_name(device),
                         c);
            TITAN_EXPECT(c.ok);
        }
    }
}

void tiny_greedy(Device device) {
    const std::filesystem::path dir = kFixtures / "tiny_llama";
    const GoldenFiles golden = load_golden(dir);
    const double tol = tolerance_of(golden);
    for (const char* name : {"tied", "untied"}) {
        HfModel hf = load_on(dir / name, device);
        const auto& prompts = golden.json.at(name).as_array();
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const std::string label = std::string("tiny ") + name + ".p" +
                                      std::to_string(i) + " " +
                                      device_name(device);
            TITAN_EXPECT(check_greedy(hf, prompts[i], tol, label).ok);
        }
    }
}

TITAN_TEST(HfGolden, TinyLogitsCpu) { tiny_logits(Device::CPU); }

TITAN_TEST(HfGolden, TinyLogitsCuda) {
    TITAN_SKIP_IF_NO_CUDA();
    tiny_logits(Device::CUDA);
}

TITAN_TEST(HfGolden, TinyGreedyCpu) { tiny_greedy(Device::CPU); }

TITAN_TEST(HfGolden, TinyGreedyCuda) {
    TITAN_SKIP_IF_NO_CUDA();
    tiny_greedy(Device::CUDA);
}

// ---------------------------------------------------------------------------
// SmolLM2-135M: the full last-position logits and 1024 sampled vocabulary ids
// at every position, all in one comparison per prompt; 32 greedy tokens.

void skip_without_smollm2() {
    if (!std::filesystem::exists(kSmolLM2Dir / "model.safetensors")) {
        TITAN_SKIP("models/SmolLM2-135M/model.safetensors is absent");
    }
}

void smollm2_logits(Device device) {
    const GoldenFiles golden = load_golden(kFixtures / "smollm2");
    const double tol = tolerance_of(golden);
    const std::vector<std::int32_t> samples =
        int_list(golden.json.at("sample_ids"));
    HfModel hf = load_on(kSmolLM2Dir, device);
    const std::int64_t vocab = hf.config.model.vocab_size;
    const auto& prompts = golden.json.at("prompts").as_array();
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const std::string p = "p" + std::to_string(i);
        const std::vector<float> logits =
            prompt_logits(*hf.model, int_list(prompts[i].at("ids")));
        const std::int64_t t = static_cast<std::int64_t>(logits.size()) / vocab;

        // ours and ref: the last position's vocab logits, then the sampled
        // ids of every position, row by row.
        std::vector<float> ours(logits.end() - vocab, logits.end());
        for (std::int64_t pos = 0; pos < t; ++pos) {
            for (const std::int32_t id : samples) {
                ours.push_back(logits[pos * vocab + id]);
            }
        }
        std::vector<float> ref = read_tensor(golden.tensors, p + ".last");
        const std::vector<float> sampled =
            read_tensor(golden.tensors, p + ".sampled");
        ref.insert(ref.end(), sampled.begin(), sampled.end());

        const LogitCheck c = compare_logits(ours, ref, tol);
        print_logits("smollm2 " + p + " " + device_name(device), c);
        TITAN_EXPECT(c.ok);
    }
}

void smollm2_greedy(Device device) {
    const GoldenFiles golden = load_golden(kFixtures / "smollm2");
    const double tol = tolerance_of(golden);
    HfModel hf = load_on(kSmolLM2Dir, device);
    const auto& prompts = golden.json.at("prompts").as_array();
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const std::string label =
            "smollm2 p" + std::to_string(i) + " " + device_name(device);
        TITAN_EXPECT(check_greedy(hf, prompts[i], tol, label).ok);
    }
}

TITAN_TEST(HfGolden, SmolLM2LogitsCpu) {
    skip_without_smollm2();
    smollm2_logits(Device::CPU);
}

TITAN_TEST(HfGolden, SmolLM2LogitsCuda) {
    TITAN_SKIP_IF_NO_CUDA();
    skip_without_smollm2();
    smollm2_logits(Device::CUDA);
}

TITAN_TEST(HfGolden, SmolLM2GreedyCpu) {
    skip_without_smollm2();
    smollm2_greedy(Device::CPU);
}

TITAN_TEST(HfGolden, SmolLM2GreedyCuda) {
    TITAN_SKIP_IF_NO_CUDA();
    skip_without_smollm2();
    smollm2_greedy(Device::CUDA);
}

}  // namespace
