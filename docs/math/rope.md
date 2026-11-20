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

- Public interface and precondition checks (rank 4, even `hd`, `pos_offset ≥ 0`): `src/ops/rope.hpp`, `src/ops/rope.cpp`. A CUDA tensor fails with `ops::rope: CUDA implementation arrives in M2` (likewise for `rope_backward`).
- Reference implementation: `src/ops/cpu/rope.cpp`; one helper rotates by `sign · a`, with `sign = +1` for the forward pass and `−1` for the backward pass.
- Tests: `tests/ops/rope_test.cpp` covers the identity at position 0, norm preservation, a known value, the pair frequencies, the position offset, the relative-position property, backward inverting forward, finite-difference gradient checks with and without an offset, and error cases.
