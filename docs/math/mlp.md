# Feed-forward blocks (SwiGLU and GELU-MLP)

## What it computes

The position-wise feed-forward network of a transformer block: it widens each token's vector from `dim` to `hidden`, applies a non-linearity and projects back to `dim`. SwiGLU (SmolLM2, Llama) gates one projection with the SiLU of another. The GELU-MLP (GPT-2) is the plain two-layer network.

## Forward

With `silu(a) = a·σ(a)` and `gelu` the tanh approximation (see [Elementwise ops](elementwise.md)), and `⊙` the elementwise product:

SwiGLU, with no biases:

`g = W_gate x`, `u = W_up x`, `a = silu(g)`, `h = a ⊙ u`, `y = W_down h`

GELU-MLP, with biases:

`p = W_fc x + b_fc`, `h = gelu(p)`, `y = W_proj h + b_proj`

## Backward derivation

Each linear layer follows [Linear](linear.md): given the gradient of its output it returns the gradient of its input and accumulates its own parameter gradients.

SwiGLU. From `dy`:

1. `dh = Linear_down.backward(dy)`.
2. `h = a ⊙ u`, so `da = dh ⊙ u` and `du = dh ⊙ a`.
3. `a = silu(g)`, so `dg = silu'(g) ⊙ da` (`silu_backward(g, da)`).
4. `g` and `u` are both linear in `x`, so `x` receives two contributions: `dx = Linear_gate.backward(dg) + Linear_up.backward(du)`.

GELU-MLP. From `dy`:

1. `dh = Linear_proj.backward(dy)`.
2. `dp = gelu'(p) ⊙ dh` (`gelu_backward(p, dh)`).
3. `dx = Linear_fc.backward(dp)`.

## Shapes

- Input `x`: `(..., dim)`; output: the same shape.
- SwiGLU: `gate_proj` and `up_proj` have weight `(hidden, dim)`, `down_proj` has `(dim, hidden)`; `g`, `u`, `a`, `h` are `(..., hidden)`.
- GELU-MLP: `fc` has weight `(hidden, dim)` and bias `(hidden)`, `proj` has weight `(dim, hidden)` and bias `(dim)`.
- Parameter names are `<name>.gate_proj.weight`, `<name>.up_proj.weight`, `<name>.down_proj.weight` and `<name>.fc.weight`, `<name>.fc.bias`, `<name>.proj.weight`, `<name>.proj.bias`.

## Mapping to code

- `src/nn/mlp.hpp`, `src/nn/mlp.cpp`: `SwiGLU` and `GeluMLP` own their `Linear` children and use `ops::silu`, `ops::gelu`, `ops::mul`, `ops::add` and the `_backward` ops.
- A training-mode forward caches `g`, `u` and `a` (SwiGLU) or `p` (GELU-MLP); the linear layers cache their own inputs.
- Tests: `tests/nn/mlp_test.cpp` gradient-checks both blocks with respect to the input and every parameter and checks the parameter names.
