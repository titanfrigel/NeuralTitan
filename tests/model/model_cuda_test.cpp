// The model on the GPU: Model::to, tied weights after the move, CPU-vs-CUDA
// parity of a full forward and backward, and overfitting one batch on CUDA.

#include <cstdint>
#include <string>
#include <vector>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include "nn/loss.hpp"
#include "nn/module.hpp"
#include "optim/adamw.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Rng;
using titan::Tensor;
using titan::model::llama_style;
using titan::model::Model;
using titan::model::ModelConfig;
using titan::nn::CrossEntropyLoss;
using titan::nn::ForwardContext;
using titan::nn::Parameter;

namespace {

std::vector<std::int32_t> random_ids(std::size_t n, std::int64_t vocab,
                                     Rng& rng) {
    std::vector<std::int32_t> ids(n);
    for (std::int32_t& id : ids) {
        id = static_cast<std::int32_t>(rng.randint(vocab));
    }
    return ids;
}

Parameter& find_parameter(Model& m, const std::string& name) {
    for (Parameter* p : m.parameters()) {
        if (p->name == name) {
            return *p;
        }
    }
    throw titan::Error("no parameter named " + name);
}

// A GPT-2 style variant: LayerNorm, GELU MLP, no RoPE, untied head.
ModelConfig gpt_style(std::int64_t vocab) {
    ModelConfig c = llama_style(vocab, 24, 2, 4, 4, 16);
    for (titan::model::BlockConfig& b : c.blocks) {
        b.mlp = "gelu";
        b.mlp_hidden = 44;
        b.norm = "layernorm";
        b.position = "none";
    }
    c.final_norm = "layernorm";
    c.tied_embeddings = false;
    return c;
}

// Builds the config twice with the same init seed, moves the second copy to
// CUDA, runs one training forward and backward on each and compares the
// logits, the loss and every parameter gradient.
void expect_model_parity(const ModelConfig& config, std::uint64_t seed) {
    Model cpu(config);
    Model gpu(config);
    Rng init_cpu(seed);
    Rng init_gpu(seed);
    cpu.init(init_cpu);
    gpu.init(init_gpu);
    gpu.to(Device::CUDA);

    const std::int64_t batch = 3;
    const std::int64_t t_len = 11;
    Rng data(seed + 1);
    const auto n = static_cast<std::size_t>(batch * t_len);
    const std::vector<std::int32_t> ids =
        random_ids(n, config.vocab_size, data);
    const std::vector<std::int32_t> targets =
        random_ids(n, config.vocab_size, data);

    // The whole model chains dozens of ops, each rounding slightly
    // differently on the two devices (reduction order, cuBLAS blocking), so
    // the bounds are those of the chained modules in module_parity_test.
    constexpr double kAtol = 1e-4;
    constexpr double kRtol = 1e-3;

    ForwardContext cpu_ctx;
    ForwardContext gpu_ctx;
    CrossEntropyLoss cpu_loss;
    CrossEntropyLoss gpu_loss;
    const Tensor cpu_logits = cpu.forward(ids, batch, t_len, cpu_ctx);
    const Tensor gpu_logits = gpu.forward(ids, batch, t_len, gpu_ctx);
    TITAN_EXPECT(gpu_logits.device() == Device::CUDA);
    TITAN_EXPECT_PARITY(cpu_logits, gpu_logits, kAtol, kRtol);
    TITAN_EXPECT_NEAR(cpu_loss.forward(cpu_logits, targets),
                      gpu_loss.forward(gpu_logits, targets), 1e-4);
    cpu.backward(cpu_loss.backward());
    gpu.backward(gpu_loss.backward());

    const std::vector<Parameter*> cpu_params = cpu.parameters();
    const std::vector<Parameter*> gpu_params = gpu.parameters();
    TITAN_EXPECT_EQ(cpu_params.size(), gpu_params.size());
    for (std::size_t i = 0; i < cpu_params.size(); ++i) {
        const titan::test::ParityResult r = titan::test::check_parity(
            cpu_params[i]->grad, gpu_params[i]->grad, kAtol, kRtol);
        if (!r.ok) {
            titan::test::report_failure(
                __FILE__, __LINE__, cpu_params[i]->name + ".grad: " + r.detail);
        }
    }
}

}  // namespace

TITAN_TEST(Model, ToCudaKeepsTiedWeights) {
    // Review focus: after the move the head and the embedding must still be
    // one tensor, or training silently differs from the CPU.
    TITAN_REQUIRE_CUDA();
    const ModelConfig config = llama_style(13, 16, 2, 2, 1, 8);
    Model cpu(config);
    Model gpu(config);
    Rng init_cpu(70);
    Rng init_gpu(70);
    cpu.init(init_cpu);
    gpu.init(init_gpu);
    TITAN_EXPECT(gpu.device() == Device::CPU);
    gpu.to(Device::CUDA);
    TITAN_EXPECT(gpu.device() == Device::CUDA);
    for (const Parameter* p : gpu.parameters()) {
        TITAN_EXPECT(p->value.device() == Device::CUDA);
        TITAN_EXPECT(p->grad.device() == Device::CUDA);
    }

    Parameter& embed = find_parameter(gpu, "embed_tokens.weight");
    TITAN_EXPECT(gpu.head_weight().value.device() == Device::CUDA);
    TITAN_EXPECT(gpu.head_weight().value.shares_storage_with(embed.value));
    TITAN_EXPECT(gpu.head_weight().grad.shares_storage_with(embed.grad));

    // One training step on each device. The CPU twin's embedding gradient
    // holds both the lookup's and the head's contribution, so matching it
    // shows the head added into the one shared CUDA gradient.
    Rng data(71);
    const std::vector<std::int32_t> ids = random_ids(2 * 6, 13, data);
    const std::vector<std::int32_t> targets = random_ids(2 * 6, 13, data);
    titan::optim::AdamWConfig cfg;
    titan::optim::AdamW cpu_opt(cpu.parameters(), cfg);
    titan::optim::AdamW gpu_opt(gpu.parameters(), cfg);
    for (Model* m : {&cpu, &gpu}) {
        titan::nn::zero_grad(m->parameters());
        ForwardContext ctx;
        CrossEntropyLoss loss;
        loss.forward(m->forward(ids, 2, 6, ctx), targets);
        m->backward(loss.backward());
    }
    // Chained ops: see expect_model_parity.
    TITAN_EXPECT_PARITY(find_parameter(cpu, "embed_tokens.weight").grad,
                        embed.grad, 1e-4, 1e-3);
    cpu_opt.step(1e-2f);
    gpu_opt.step(1e-2f);
    TITAN_EXPECT(gpu.head_weight().value.shares_storage_with(embed.value));
    TITAN_EXPECT_PARITY(find_parameter(cpu, "embed_tokens.weight").value,
                        gpu.head_weight().value, 1e-4, 1e-3);

    // Moving back keeps the tie as well.
    gpu.to(Device::CPU);
    TITAN_EXPECT(gpu.device() == Device::CPU);
    TITAN_EXPECT(gpu.head_weight().value.shares_storage_with(
        find_parameter(gpu, "embed_tokens.weight").value));
}

TITAN_TEST(Model, CudaForwardBackwardMatchesCpu) {
    TITAN_REQUIRE_CUDA();
    expect_model_parity(llama_style(37, 24, 2, 4, 2, 16), 72);
    expect_model_parity(gpt_style(37), 73);
}

TITAN_TEST(Model, CudaOverfitsOneBatch) {
    // NFR-3 on the GPU, with the config and step budget of
    // Model.OverfitsSingleBatch.
    TITAN_REQUIRE_CUDA();
    const std::int64_t batch = 4;
    const std::int64_t t_len = 16;
    const std::int64_t vocab = 20;
    Model m(llama_style(vocab, 32, 2, 4, 2, 16));
    Rng rng(62);
    m.init(rng);
    m.to(Device::CUDA);
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> targets;
    for (std::int64_t b = 0; b < batch; ++b) {
        const std::vector<std::int32_t> seq =
            random_ids(static_cast<std::size_t>(t_len + 1), vocab, rng);
        ids.insert(ids.end(), seq.begin(), seq.end() - 1);
        targets.insert(targets.end(), seq.begin() + 1, seq.end());
    }

    // Constructed after the move, so the moments live on the GPU.
    titan::optim::AdamWConfig cfg;
    titan::optim::AdamW opt(m.parameters(), cfg);
    CrossEntropyLoss loss;
    float value = 0.0f;
    for (int step = 0; step < 300; ++step) {
        titan::nn::zero_grad(m.parameters());
        ForwardContext ctx;
        value = loss.forward(m.forward(ids, batch, t_len, ctx), targets);
        if (value < 0.05f) {
            break;
        }
        m.backward(loss.backward());
        opt.step(3e-3f);
    }
    TITAN_EXPECT(value < 0.05f);
}
