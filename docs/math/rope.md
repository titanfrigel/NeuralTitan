# Rotary position embedding (RoPE)

## What it computes

Injects position into attention by rotating the query and key vectors. Each head vector is split into pairs of coordinates, and each pair is rotated in its own 2-D plane by an angle proportional to the token position. Because rotations compose, the dot product between a rotated query and a rotated key depends only on the distance between their positions, which is the relative position information attention needs.

## Forward

Let `hd` be the head dimension (even), `p = pos_offset + t` the position of token `t`, and `i < hd/2` the pair index. The pair is `(x_i, x_{i+hd/2})` and its angle is

`a = p · theta^(−2i/hd)`

so low pairs rotate fast and high pairs slowly. The rotation is the 2-D matrix:

`y_i = x_i · cos a − x_{i+hd/2} · sin a`
`y_{i+hd/2} = x_{i+hd/2} · cos a + x_i · sin a`

**Layout.** This is the half-split ("rotate_half") layout: pair `i` is made of coordinates `i` and `i + hd/2`, not of adjacent coordinates `2i` and `2i+1`. Hugging Face Llama-family checkpoints, including SmolLM2, are trained with this layout and their query and key projection weights are arranged for it, so using it lets the library reproduce their outputs exactly.

**Relative position.** Write a pair as a complex number `z = x_i + i·x_{i+hd/2}`. The rotation is `z ↦ z · e^{i a}`. For a query `q` at position `m` and a key `k` at position `n`, the pair's contribution to the dot product is

`Re( (q e^{i m ω}) · conj(k e^{i n ω}) ) = Re( q · conj(k) · e^{i (m−n) ω} )`

with `ω = theta^(−2i/hd)`. It depends on `m` and `n` only through `m − n`; summing over pairs gives the full dot product.

The angle, its cosine and sine are computed in `double` and applied to the `float` values. Position 0 gives `a = 0`, so the rotation is the identity.

## Backward derivation

The forward map is linear in `x`: `y = R x`, where `R` is block diagonal with one 2×2 rotation per pair,

`R_p = [[cos a, −sin a], [sin a, cos a]]`.

Then `dx = Rᵀ dy`. A rotation matrix is orthogonal, so `Rᵀ = R⁻¹`, which is the rotation by `−a`: `cos(−a) = cos a`, `sin(−a) = −sin a`. Hence

`dx_i = dy_i · cos a + dy_{i+hd/2} · sin a`
`dx_{i+hd/2} = dy_{i+hd/2} · cos a − dy_i · sin a`

There are no parameters, and `pos_offset` and `theta` are constants, so there is nothing else to differentiate.

## Shapes

- `rope(x, pos_offset, theta)`: `x` is `(B, H, T, hd)` with `hd` even and `pos_offset ≥ 0`; the output has the same shape.
- `rope_backward(dy, pos_offset, theta)`: `dy` has the shape of `x`; the result has the same shape.

## Mapping to code

- Public interface and precondition checks (rank 4, even `hd`, `pos_offset ≥ 0`): `src/ops/rope.hpp`, `src/ops/rope.cpp`. The dispatch routes by the tensor's device to the CPU or the CUDA implementation.
- Reference implementation: `src/ops/cpu/rope.cpp`; one helper rotates by `sign · a`, with `sign = +1` for the forward pass and `−1` for the backward pass.
- CUDA kernel: `src/ops/cuda/rope.cu`, declared in `src/ops/cuda/rope.hpp`. `tests/ops/cuda/rope_parity_test.cpp` compares `rope` and `rope_backward` with the CPU for `pos_offset` 0 and 7 and head dimensions 2 and 64.
- Tests: `tests/ops/rope_test.cpp` covers the identity at position 0, norm preservation, a known value, the pair frequencies, the position offset, the relative-position property, backward inverting forward, finite-difference gradient checks with and without an offset, and error cases.

## CUDA implementation

The kernel is in `src/ops/cuda/rope.cu`. It has no reduction and no atomic, so the result is deterministic. See [CUDA execution](cuda-execution.md) for the grid and block vocabulary.

- Thread mapping: one thread per rotation pair, in a grid-stride loop over the `B · H · T · hd/2` pairs. Pair `k` is row `k / (hd/2)` (a `(batch, head, position)` triple) and pair index `i = k mod (hd/2)`; the token position is `t = row mod T`. The thread reads the two coordinates `x[row·hd + i]` and `x[row·hd + i + hd/2]` and writes the two rotated values to the same places of the output. Each element is read and written by exactly one thread.
- Angle: as on the CPU, `a = (pos_offset + t) · theta^(−2i/hd)` is computed in `double` from `pow`, and `cos a` and `sin a` are taken in `double` and rounded to `float` before they multiply the data. `pos_offset` enters only through the position `p`.
- Backward: the same kernel with `sign = −1`, so `sin` is negated, which is the inverse rotation derived above.

Differences from the CPU:

- The device `pow`, `sin` and `cos` in `double` can differ from the host library by an ulp of a `double`. After rounding to `float` this almost never changes a value, and when it does the change is one `float` ulp, far inside the parity tolerance (`atol = 1e-5`, `rtol = 1e-4`).
- The CUDA compiler may fuse `lo · c − hi · s` into fused multiply-adds, which changes the rounding of the products by at most one `float` ulp.
