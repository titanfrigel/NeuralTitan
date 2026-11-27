#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "harness/test.hpp"
#include "io/file.hpp"
#include "io/json.hpp"
#include "io/safetensors.hpp"

// Checks that the golden fixtures written by tools/make_llama_fixtures.py are
// readable by our own JSON parser and safetensors reader and agree with each
// other: every tensor that golden.json implies exists, in F32, with the
// implied shape.

namespace {

using titan::io::JsonValue;
using titan::io::SafetensorsDtype;
using titan::io::SafetensorsFile;

const std::filesystem::path kFixtures = TITAN_FIXTURES_DIR;

void expect_tensor(const SafetensorsFile& file, const std::string& name,
                   const std::vector<std::int64_t>& shape) {
    const auto* entry = file.find(name);
    TITAN_EXPECT(entry != nullptr);
    if (entry == nullptr) {
        return;
    }
    TITAN_EXPECT(entry->dtype == SafetensorsDtype::F32);
    TITAN_EXPECT(entry->shape == shape);
}

std::int64_t count(const JsonValue& v) {
    return static_cast<std::int64_t>(v.as_array().size());
}

// Checks the shape of a prompt's token lists.
void expect_greedy_fields(const JsonValue& prompt, std::int64_t new_tokens) {
    const auto tokens = count(prompt.at("greedy"));
    const auto gaps = count(prompt.at("greedy_gaps"));
    TITAN_EXPECT(tokens >= 1 && tokens <= new_tokens);
    // One gap per generated step; a final end-of-text token has a gap too.
    TITAN_EXPECT(gaps == tokens || gaps == tokens + 1);
    TITAN_EXPECT(count(prompt.at("ids")) >= 1);
}

TITAN_TEST(Golden, FixturesReadable) {
    // SmolLM2-135M.
    {
        const auto golden = titan::io::parse_json(
            titan::io::read_file(kFixtures / "smollm2" / "golden.json"));
        const auto file =
            SafetensorsFile::open(kFixtures / "smollm2" / "golden.safetensors");
        TITAN_EXPECT_EQ(
            golden.at("revision").as_string(),
            std::string("93efa2f097d58c2a74874c7e644dbc9b0cee75a2"));
        TITAN_EXPECT_EQ(golden.at("tolerance").as_number(), 1e-3);
        const auto& samples = golden.at("sample_ids").as_array();
        TITAN_EXPECT_EQ(samples.size(), std::size_t{1024});
        std::int64_t previous = -1;
        for (const auto& s : samples) {
            const auto id = s.as_int();
            TITAN_EXPECT(id > previous && id < 49152);  // sorted, in range
            previous = id;
        }
        const auto& prompts = golden.at("prompts").as_array();
        TITAN_EXPECT_EQ(prompts.size(), std::size_t{4});
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const auto& prompt = prompts[i];
            expect_greedy_fields(prompt, 32);
            const auto t = count(prompt.at("ids"));
            const std::string p = "p" + std::to_string(i);
            expect_tensor(file, p + ".last", {49152});
            expect_tensor(file, p + ".sampled", {t, 1024});
            expect_tensor(file, p + ".top2_gap", {t});
        }
        TITAN_EXPECT_EQ(file.entries().size(), prompts.size() * 3);
    }
    // The tiny models.
    {
        const auto golden = titan::io::parse_json(
            titan::io::read_file(kFixtures / "tiny_llama" / "golden.json"));
        const auto file = SafetensorsFile::open(kFixtures / "tiny_llama" /
                                                "golden.safetensors");
        TITAN_EXPECT_EQ(golden.at("tolerance").as_number(), 1e-3);
        std::size_t tensors = 0;
        for (const char* name : {"tied", "untied"}) {
            const auto& prompts = golden.at(name).as_array();
            TITAN_EXPECT_EQ(prompts.size(), std::size_t{3});
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                expect_greedy_fields(prompts[i], 16);
                const auto t = count(prompts[i].at("ids"));
                const std::string p =
                    std::string(name) + ".p" + std::to_string(i);
                expect_tensor(file, p + ".logits", {t, 512});
                expect_tensor(file, p + ".top2_gap", {t});
                tensors += 2;
            }
        }
        TITAN_EXPECT_EQ(file.entries().size(), tensors);
    }
}

}  // namespace
