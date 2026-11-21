# AdamW, gradient clipping and the learning-rate schedule

## What it computes

Turns the gradients of one training step into a parameter update. AdamW keeps a running average of each gradient (the first moment) and of its square (the second moment), and moves each parameter by the averaged gradient divided by the root of the averaged square, so every parameter takes steps of a similar size whatever its gradient scale. Weight decay shrinks the weights separately from the gradient step. Gradient clipping bounds the size of one step's gradient, and the schedule chooses the learning rate for each step.

## Forward

For a parameter `p` with gradient `g` at step `t = 1, 2, …`:

- `m_t = β₁ m_{t−1} + (1 − β₁) g`
- `v_t = β₂ v_{t−1} + (1 − β₂) g²`
- `m̂ = m_t / (1 − β₁ᵗ)`, `v̂ = v_t / (1 − β₂ᵗ)` (bias correction)
- `p ← p − lr · ( m̂ / (√v̂ + ε) + λ p )`

`m₀ = v₀ = 0`. Because the moments start at zero they are biased towards zero in the first steps: with a constant gradient, `m_t = (1 − β₁ᵗ) g`. Dividing by `1 − β₁ᵗ` removes the bias exactly; at `t = 1` it gives `m̂ = g` and `v̂ = g²`.

Weight decay is decoupled: the term `λ p` is added outside the adaptive ratio. Folding it into the gradient as L2 regularization (`g + λ p`) would pass it through `m̂ / √v̂`, so parameters with a large gradient history would be decayed less, and the decay would depend on the gradient statistics. Decoupled, every parameter shrinks by the same factor `1 − lr·λ` per step.

Parameters of rank 1 (norm weights and biases) get `λ = 0`: decaying a scale that should stay near 1, or a bias, fights the model without a regularization benefit. This is the common practice, applied by shape.

**Gradient clipping.** With all gradients flattened into one vector, the global norm is `‖g‖ = √(Σ_params Σ_i g_i²)`, accumulated in double. If `‖g‖ > c`, every gradient is multiplied by `c / ‖g‖`, which keeps the direction and gives the new global norm `c`. Otherwise nothing changes. The norm before clipping is returned. A non-finite norm is returned without scaling; the trainer decides what to do.

**Schedule.** With warmup `W`, total steps `S`, steps counted from 0:

- `step < W`: `lr = max_lr · (step + 1) / W`, a linear ramp that reaches `max_lr` at `step = W − 1`.
- `W ≤ step < S`: `progress = (step − W) / max(1, S − 1 − W)` and `lr = min_lr + ½ (max_lr − min_lr) (1 + cos(π · progress))`, which falls from `max_lr` at `step = W` to `min_lr` at `step = S − 1`.
- `step ≥ S`: `lr = min_lr`.

## Backward derivation

The optimizer has no gradient of its own: it consumes the gradients computed by the modules. The update follows from the definitions above. The first moment is an exponential moving average, `m_t = (1 − β₁) Σ_k β₁^{t−k} g_k`, so for a constant gradient its weights sum to `1 − β₁ᵗ`, which is the bias-correction divisor. The same holds for `v_t` with `β₂`. The clip scaling is linear, so after clipping `Σ (c/‖g‖ · g_i)² = c²`.

## Shapes

- `AdamW`: one first moment and one second moment per parameter, each with the parameter's shape.
- `clip_grad_norm`: takes any list of parameters, returns a scalar `double`.
- `lr_at`: scalar in, scalar out; requires `warmup ≥ 1` and `total > warmup`.

## Mapping to code

- `src/optim/adamw.hpp`, `adamw.cpp`: `optim::AdamW`. `step(lr)` increments the counter and calls `ops::adamw_update` per parameter, with `weight_decay = 0` when the parameter has rank 1.
- `src/ops/adamw.hpp`: `ops::adamw_update`, the update formula. `src/ops/adamw.cpp` validates and routes to `src/ops/cpu/adamw.cpp` or `src/ops/cuda/adamw.cu`.
- `src/optim/clip.hpp`, `clip.cpp`: `optim::clip_grad_norm`, using `ops::sum_squares` and `ops::scale_`.
- `src/optim/schedule.hpp`, `schedule.cpp`: `optim::lr_at`.
- `tests/optim/optim_test.cpp`: the tests.

## CUDA implementation

`src/ops/cuda/adamw.cu` runs the update of the Forward section on CUDA tensors, in one kernel launch on the default stream.

- **Thread mapping.** One thread per element, in a grid-stride loop: thread `t` handles elements `t`, `t + stride`, `t + 2·stride`, … The update of one element reads and writes only that element of `param`, `m` and `v` and reads that element of `grad`, so no thread depends on another and there is no reduction.
- **Bias correction on the host.** `1 − β₁ᵗ` and `1 − β₂ᵗ` are the same for every element, so the host computes them once in double with `std::pow` and passes them to the kernel, which only divides by them.
- **Precision.** Like the CPU reference, each element is computed in double and rounded to float when stored back. Only the double `sqrt` and the basic arithmetic run on the device, which are correctly rounded, so CPU and CUDA agree to the last rounding of the float result. The parity test uses the default tolerances.
- **Determinism.** There are no atomics and no reduction, so a repeated run gives bit-identical results.
- **Differences from the CPU.** None in the formula. Weight decay stays decoupled and `ε` is added to `√v̂`, as above. The dispatch requires `param`, `grad`, `m` and `v` to be on the same device.
- **Test.** `AdamWParity.TenSteps` in `tests/ops/cuda/adamw_parity_test.cpp` runs 10 steps on both devices and compares `param`, `m` and `v`.
