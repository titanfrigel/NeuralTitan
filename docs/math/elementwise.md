# Elementwise ops

## What it computes

Operations applied independently to each element: addition, multiplication, scaling, and the two activation functions SiLU and GELU. Activations add the non-linearity that lets stacked linear layers model more than a single linear map.

## Forward

- `add`: `y = a + b`; `add_` computes `a ← a + b` in place.
- `add_rowwise_`: `y[r][j] ← y[r][j] + b[j]` for every row `r`, in place; the broadcast add of a bias over the last dimension.
- `mul`: `y = a ⊙ b`.
- `scale`: `y = s·x`; `scale_` is the in-place form `x ← s·x`.
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

`add_rowwise_` takes `y` of any shape with last dimension `D` and `b` of shape `(D)`. Other two-input ops take two tensors of identical shape and return that shape. `scale`, `silu` and `gelu` map a tensor to a tensor of the same shape. `silu_backward` and `gelu_backward` take `x` and `dy` of the same shape and return `dx` of that shape.

## CUDA implementation

The execution model is explained in [CUDA execution](cuda-execution.md).

- **Thread mapping.** Every op is one element-parallel kernel: a grid of `grid_for(n, 256)` blocks of 256 threads runs a grid-stride loop, and each iteration computes one output element `i` from input element `i` alone. `add_rowwise_` reads the bias at `i mod D`, the column of element `i` in a row-major tensor with last dimension `D`, so it needs no second grid dimension for rows. The in-place ops write each element from the one thread that read it, so no thread sees another's write.
- **Reduction order.** None: no op combines elements, so the results are deterministic without any ordering rule.
- **What differs from the CPU.** The formulas are the same term for term, with the same float constants. Two things can change the last bits: the device `expf` and `tanhf` are accurate to a few units in the last place but are not the host library's functions, and nvcc fuses a multiply and an add into one fused multiply-add (one rounding instead of two) where the CPU build rounds twice. The parity tests therefore compare with `|c − g| ≤ 1e-5 + 1e-4·|c|`, the default tolerance, which every op meets.

## Mapping to code

- Public interface and precondition checks: `src/ops/elementwise.hpp`, `src/ops/elementwise.cpp`. `detail::route` (`src/ops/detail.hpp`) picks the backend and rejects tensors on different devices with `ops::<name>: tensors are on different devices (cpu vs cuda)`.
- CPU reference loops: `src/ops/cpu/elementwise.cpp`, one function per formula above.
- CUDA kernels: `src/ops/cuda/elementwise.cu`, one kernel per formula, with the plain C++ interface in `src/ops/cuda/elementwise.hpp`.
- Tests: `tests/ops/elementwise_test.cpp` checks values and compares both activation gradients with central differences using `tests/support/gradcheck.hpp`. `tests/ops/cuda/elementwise_parity_test.cpp` compares every op on both devices for a single element, a `37 × 129` tensor and a `1024 × 1025` tensor, and pins the device-mismatch message.
