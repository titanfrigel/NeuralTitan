#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

#include "core/device.hpp"
#include "io/json.hpp"
#include "model/model.hpp"
#include "optim/adamw.hpp"
#include "runtime/data_loader.hpp"

namespace titan::runtime {

// The "train" section of a config file. Sizes are in sequences and tokens;
// one optimizer step consumes grad_accum micro-batches of batch x seq tokens.
struct TrainConfig {
    std::int64_t batch = 0, seq = 0;
    std::int64_t steps = 0;
    std::int64_t grad_accum = 1;
    float lr = 0.0f, min_lr = 0.0f;
    std::int64_t warmup = 0;
    float weight_decay = 0.1f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f;
    float grad_clip = 1.0f;
    std::int64_t eval_interval = 0, eval_batches = 0;
    std::int64_t log_interval = 0;
    std::uint64_t seed = 0;
    // Share of the corpus held out for validation; used by the caller that
    // splits the tokens, not by Trainer itself.
    float val_fraction = 0.1f;
    // Where the model trains. Set by the caller, not read from the config
    // file; the Trainer moves the model there before the first step.
    Device device = Device::CPU;
};

// Parses and validates the "train" section. Required: batch, seq, steps, lr,
// min_lr, warmup, eval_interval, eval_batches, log_interval, seed. Optional
// (default): grad_accum (1), grad_clip (1.0), val_fraction (0.1), beta1 (0.9),
// beta2 (0.95), eps (1e-8), weight_decay (0.1).
//
// Throws titan::Error naming the field path, e.g. "train.lr", for a missing or
// unknown key, a wrong type, a non-positive size, lr <= 0, min_lr < 0 or
// min_lr > lr, a beta outside [0, 1), eps or grad_clip <= 0, weight_decay < 0,
// val_fraction outside (0, 1), or warmup outside [1, steps).
TrainConfig parse_train_config(const io::JsonValue& train_section);

struct TrainResult {
    // Loss of the last optimizer step (mean over its micro-batches) and the
    // last validation loss.
    float final_train_loss = 0.0f, final_val_loss = 0.0f;
    double seconds = 0.0;
    // Training tokens processed: steps * grad_accum * batch * seq.
    std::int64_t tokens = 0;
};

// Trains a model on a token stream with AdamW, gradient accumulation, global
// gradient-norm clipping and the warmup + cosine schedule, logging progress
// and validation loss. Steps are numbered from 1 in all output.
class Trainer {
public:
    // The model must outlive the trainer and be initialised (on the CPU, so
    // that one seed gives the same weights on every device). The constructor
    // moves it to config.device, then creates the optimizer state there; the
    // model stays on that device afterwards. Throws
    // titan::Error unless seq <= model.config().context_length and each token
    // stream holds at least seq + 2 tokens. The validation loader is seeded
    // with seed + 1.
    Trainer(model::Model& model, TrainConfig config,
            std::vector<std::int32_t> train_tokens,
            std::vector<std::int32_t> val_tokens, std::ostream& log);

    // Runs config.steps optimizer steps. Every log_interval steps writes
    //   step 200/2000 loss 2.1034 lr 9.8e-04 grad_norm 0.512 tok/s 12345
    // (tok/s counts the tokens since the previous log line), and every
    // eval_interval steps and after the last step
    //   eval step 200 val_loss 2.0871
    // Throws titan::Error "training diverged at step N: loss=<v> grad_norm=<v>"
    // when the loss or the gradient norm is not finite; the parameters are
    // not updated by that step.
    TrainResult run();

    // Test seam: the training loader, so a test can make it sequential.
    DataLoader& train_loader() { return train_loader_; }

private:
    float evaluate();

    model::Model& model_;
    TrainConfig config_;
    DataLoader train_loader_;
    DataLoader val_loader_;
    std::ostream& log_;
    optim::AdamW optimizer_;
};

}  // namespace titan::runtime
