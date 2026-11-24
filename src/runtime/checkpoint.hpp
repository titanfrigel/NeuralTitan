#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "io/json.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "optim/adamw.hpp"
#include "runtime/trainer.hpp"
#include "tokenizer/tokenizer.hpp"

namespace titan::runtime {

// A self-contained training checkpoint: everything `titan train --resume` and
// `titan generate` need, stored in one io checkpoint file
// (io::write_checkpoint_file). The meta object of the file is
//   {"vocab_size": V,
//    "model": model_config_to_json(model),
//    "train": train_config_to_json(train),
//    "tokenizer": <the tokenizer's JSON>,
//    "state": {"step", "optimizer_steps", "train_rng", "val_rng",
//              "train_tokens", "val_tokens", "train_crc32", "val_crc32"}}
// and its tensors are every parameter under its name, in the model's
// parameter order, then "optim.m.<name>" and "optim.v.<name>" for each.
struct Checkpoint {
    model::ModelConfig model;
    TrainConfig train;
    io::JsonValue tokenizer;
    TrainerState state;
};

// Writes `model`'s parameters, `ckpt.state.optimizer`'s moments (one m and one
// v per parameter, in the model's parameter order) and the meta above to
// `path`, atomically (see io::write_checkpoint_file). The parameters may be on
// any device; they are copied to the CPU. Throws titan::Error when the number
// of moments or a moment's shape does not match the parameters, or on an I/O
// failure.
void save_checkpoint(const std::filesystem::path& path, model::Model& model,
                     const Checkpoint& ckpt);

// Reads a checkpoint written by save_checkpoint, treating the file as
// untrusted: the meta is validated like a config file, and the train and
// model sections go through parse_train_config and parse_model_config.
//
// With a model, reads the whole file and copies every parameter onto the
// model's current device, and fills state.optimizer.m and .v (on the CPU, in
// the model's parameter order). It throws titan::Error naming the file and
// the tensor for a missing tensor, an extra tensor, or a shape mismatch (with
// both shapes), and naming the file and the field (e.g. "model.rope_theta")
// when the saved model config differs from model->config(); the model is
// unchanged then. The model's parameters are
// replaced in place, so a Trainer or an optimizer that holds them stays
// valid.
//
// With nullptr, reads only the header (io::read_checkpoint_header), so the
// caller can build the model from Checkpoint::model first. Before returning
// it bounds that config by the file: model::parameter_count(model) times
// three (the parameters and both moments) must equal the values the file's
// tensor entries hold, else titan::Error names the file and both counts. A
// header that claims a huge model therefore fails before any allocation.
// The result's state.optimizer.m and .v are empty.
Checkpoint load_checkpoint(const std::filesystem::path& path,
                           model::Model* model);

// The tokenizer stored in `ckpt` (read from `path`), rebuilt with
// tokenizer::load_tokenizer (a CharTokenizer or a BpeTokenizer). Throws
// titan::Error naming the file for a malformed or unsupported tokenizer, or
// when its vocabulary size differs from ckpt.model.vocab_size, the size the
// model was built with.
std::unique_ptr<tokenizer::Tokenizer> checkpoint_tokenizer(
    const std::filesystem::path& path, const Checkpoint& ckpt);

}  // namespace titan::runtime
