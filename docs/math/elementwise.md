# Elementwise ops

## What it computes

Operations applied independently to each element: addition, multiplication, scaling, and the two activation functions SiLU and GELU. Activations add the non-linearity that lets stacked linear layers model more than a single linear map.

## Forward

- `add`: `y = a + b`; `add_` computes `a ← a + b` in place.
- `mul`: `y = a ⊙ b`.
- `scale`: `y = s·x`.
- `silu`: `y = x·σ(x)`, where `σ(x) = 1 / (1 + e^(−x))`.
- `gelu` (tanh approximation): `y = 0.5·x·(1 + tanh(u))`, where `u = k·(x + 0.044715·x³)` and `k = √(2/π)`.

## Backward derivation

Let `dy = ∂L/∂y`. For an elementwise `y = f(x)` the chain rule gives `dx = dy ⊙ f'(x)`, so only `f'` is needed.

- `add`: `∂y/∂a = ∂y/∂b = 1`, so `da = db = dy`.
- `mul`: `da = dy ⊙ b`, `db = dy ⊙ a`.
- `scale`: `dx = s·dy`.
- `silu`: with `σ = σ(x)`, use `σ' = σ(1 − σ)`. Then
  `f' = σ + x·σ(1 − σ) = σ·(1 + x·(1 − σ))`.
- `gelu`: with `t = tanh(u)`, use `t' = (1 − t²)·u'` and `u' = k·(1 + 3·0.044715·x²)`. The product rule on `0.5·x·(1 + t)` gives
  `f' = 0.5·(1 + t) + 0.5·x·(1 − t²)·k·(1 + 3·0.044715·x²)`.

## Shapes

Two-input ops take two tensors of identical shape and return that shape. `scale`, `silu` and `gelu` map a tensor to a tensor of the same shape. `silu_backward` and `gelu_backward` take `x` and `dy` of the same shape and return `dx` of that shape.

## Mapping to code

- Public interface and precondition checks: `src/ops/elementwise.hpp`, `src/ops/elementwise.cpp`. A CUDA tensor fails with `ops::<name>: CUDA implementation arrives in M2` (`src/ops/detail.hpp`).
- CPU reference loops: `src/ops/cpu/elementwise.cpp`, one function per formula above.
- Tests: `tests/ops/elementwise_test.cpp` checks values and compares both activation gradients with central differences using `tests/support/gradcheck.hpp`.
