#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
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
    // Save a checkpoint every this many steps; 0 disables periodic
    // checkpoints.
    std::int64_t checkpoint_interval = 0;

    bool operator==(const TrainConfig&) const = default;
};

// Parses and validates the "train" section. Required: batch, seq, steps, lr,
// min_lr, warmup, eval_interval, eval_batches, log_interval, seed. Optional
// (default): grad_accum (1), grad_clip (1.0), val_fraction (0.1), beta1 (0.9),
// beta2 (0.95), eps (1e-8), weight_decay (0.1), checkpoint_interval (0).
//
// Throws titan::Error naming the field path, e.g. "train.lr", for a missing or
// unknown key, a wrong type, a non-positive size, lr <= 0, min_lr < 0 or
// min_lr > lr, a beta outside [0, 1), eps or grad_clip <= 0, weight_decay < 0,
// val_fraction outside (0, 1), checkpoint_interval < 0, or warmup outside
// [1, steps).
TrainConfig parse_train_config(const io::JsonValue& train_section);

// The "train" section for `c`: every field except `device`, which is not part
// of the file format. parse_train_config(train_config_to_json(c)) equals `c`
// with the default device.
io::JsonValue train_config_to_json(const TrainConfig& c);

struct TrainResult {
    // Loss of the last optimizer step (mean over its micro-batches) and the
    // last validation loss, or 0 when the call ran no such step or
    // evaluation.
    float final_train_loss = 0.0f, final_val_loss = 0.0f;
    double seconds = 0.0;
    // Training tokens processed by this call: the steps it ran times
    // grad_accum * batch * seq.
    std::int64_t tokens = 0;
};

// Where a training run stands after a completed step: everything besides the
// model's parameters and the configs that a resumed run needs to continue bit
// for bit. The token counts let a resume detect that the token files changed.
struct TrainerState {
    // The last completed step, numbered from 1; 0 before the first.
    std::int64_t step = 0;
    optim::AdamWState optimizer;
    // Rng::save_state of the training and the validation loader.
    std::string train_rng, val_rng;
    std::int64_t train_tokens = 0, val_tokens = 0;
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

    // Resumes a run when `resume` holds a state (Trainer::state() of the
    // interrupted run, usually from load_checkpoint). The model must already
    // hold that run's parameters, on any device: as above, the constructor
    // moves the model to config.device first, then creates the optimizer
    // there and loads resume->optimizer into it, so the moments land on the
    // parameters' device. The loaders' generators are restored and run()
    // starts at step resume->step + 1. Throws titan::Error
    // "resume: train tokens changed (checkpoint N, now M)" (or "val tokens")
    // when a token stream's length differs from the saved count, and for a
    // step outside [0, steps], an optimizer step count different from the
    // step, or a malformed generator state. Without a state, it is the
    // constructor above.
    Trainer(model::Model& model, TrainConfig config,
            std::vector<std::int32_t> train_tokens,
            std::vector<std::int32_t> val_tokens, std::ostream& log,
            std::optional<TrainerState> resume);

    // Runs the optimizer steps after the last completed one (state().step)
    // up to config.steps, or up to and including step stop_after when given;
    // none when every step is done.
    // Every log_interval steps writes
    //   step 200/2000 loss 2.1034 lr 9.8e-04 grad_norm 0.512 tok/s 12345
    // (tok/s counts the tokens since the previous log line), and every
    // eval_interval steps and after step config.steps
    //   eval step 200 val_loss 2.0871
    // A stopped run does not evaluate at stop_after unless eval_interval
    // says so, so that it stays bit-identical to an uninterrupted one.
    // Calls the on_checkpoint callback, after the step's logging and
    // evaluation, every checkpoint_interval steps (when > 0), after step
    // config.steps and after step stop_after, at most once per step.
    // Throws titan::Error "training diverged at step N: loss=<v> grad_norm=<v>"
    // when the loss or the gradient norm is not finite; the parameters are
    // not updated by that step. Throws unless state().step < stop_after <=
    // config.steps.
    TrainResult run(std::optional<std::int64_t> stop_after = {});

    // Registers the function called at each checkpoint step (see run()) with
    // that step; it typically saves state() and the model. Replaces any
    // earlier callback.
    void on_checkpoint(std::function<void(std::int64_t step)> callback);

    // The state after the last completed step, with the optimizer moments
    // copied to the CPU.
    TrainerState state() const;

    // Test seam: the training loader, so a test can make it sequential.
    DataLoader& train_loader() { return train_loader_; }

private:
    // The checks and restores of the resuming constructor.
    void resume_from(const TrainerState& s);
    float evaluate();

    model::Model& model_;
    TrainConfig config_;
    std::int64_t train_tokens_, val_tokens_;
    DataLoader train_loader_;
    DataLoader val_loader_;
    std::ostream& log_;
    optim::AdamW optimizer_;
    std::int64_t step_ = 0;
    std::function<void(std::int64_t)> on_checkpoint_;
};

}  // namespace titan::runtime
