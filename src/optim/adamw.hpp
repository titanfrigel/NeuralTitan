#pragma once

#include <cstdint>
#include <vector>

#include "core/tensor.hpp"
#include "nn/module.hpp"

namespace titan::optim {

struct AdamWConfig {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 1e-8f;
    float weight_decay = 0.1f;
};

// The optimizer state: the step counter and, per parameter in the order of the
// constructor's `params`, the first and second moments. The tensors are on the
// CPU.
struct AdamWState {
    std::int64_t steps = 0;
    std::vector<Tensor> m, v;
};

// AdamW with decoupled weight decay (docs/math/adamw.md). Keeps a first and a
// second moment per parameter, zero-filled at construction. Parameters of rank
// 1 (norm weights, biases) are not decayed. The parameters must outlive the
// optimizer.
class AdamW {
public:
    AdamW(std::vector<nn::Parameter*> params, AdamWConfig config);

    // One update of every parameter with learning rate `lr`. The step counter
    // is incremented first, so the first update uses step 1.
    void step(float lr);

    std::int64_t steps() const { return steps_; }

    // A deep copy of the state, moved to the CPU.
    AdamWState state() const;

    // Replaces the state with `s`, copying the moments to each parameter's
    // device. Throws titan::Error naming the parameter when the number of
    // moments or a shape does not match; the optimizer is unchanged then.
    void load_state(const AdamWState& s);

private:
    std::vector<nn::Parameter*> params_;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
    AdamWConfig config_;
    std::int64_t steps_ = 0;
};

}  // namespace titan::optim
