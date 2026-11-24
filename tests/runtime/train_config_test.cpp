#include <cstdint>
#include <string>

#include "core/error.hpp"
#include "harness/test.hpp"
#include "io/json.hpp"
#include "runtime/trainer.hpp"

using titan::runtime::parse_train_config;
using titan::runtime::train_config_to_json;
using titan::runtime::TrainConfig;

namespace {

constexpr const char* kRequired = R"({
    "batch": 4, "seq": 8, "steps": 100, "lr": 0.001, "min_lr": 0.0001,
    "warmup": 10, "eval_interval": 20, "eval_batches": 2,
    "log_interval": 5, "seed": 7
})";

}  // namespace

TITAN_TEST(TrainConfig, JsonRoundTrip) {
    TrainConfig c;
    c.batch = 3;
    c.seq = 17;
    c.steps = 250;
    c.grad_accum = 4;
    c.lr = 3e-4f;
    c.min_lr = 3e-5f;
    c.warmup = 25;
    c.weight_decay = 0.01f;
    c.beta1 = 0.85f;
    c.beta2 = 0.999f;
    c.eps = 1e-6f;
    c.grad_clip = 0.5f;
    c.eval_interval = 50;
    c.eval_batches = 3;
    c.log_interval = 10;
    c.seed = 123456789012345ull;
    c.val_fraction = 0.05f;
    c.checkpoint_interval = 100;
    const std::string text = titan::io::to_json(train_config_to_json(c));
    TITAN_EXPECT(parse_train_config(titan::io::parse_json(text)) == c);

    // The device is not part of the file format.
    TrainConfig on_gpu = c;
    on_gpu.device = titan::Device::CUDA;
    TITAN_EXPECT(train_config_to_json(on_gpu).find("device") == nullptr);
}

TITAN_TEST(TrainConfig, CheckpointIntervalOptionalAndNonNegative) {
    titan::io::JsonValue j = titan::io::parse_json(kRequired);
    TITAN_EXPECT_EQ(parse_train_config(j).checkpoint_interval, std::int64_t{0});

    j.set("checkpoint_interval", titan::io::JsonValue::number(25.0));
    TITAN_EXPECT_EQ(parse_train_config(j).checkpoint_interval,
                    std::int64_t{25});

    j.set("checkpoint_interval", titan::io::JsonValue::number(-1.0));
    std::string message;
    try {
        parse_train_config(j);
    } catch (const titan::Error& e) {
        message = e.what();
    }
    TITAN_EXPECT(message.find("train.checkpoint_interval") !=
                 std::string::npos);
}
