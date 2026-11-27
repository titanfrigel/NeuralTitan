# Hugging Face Llama weights

How NeuralTitan reads a Hugging Face Llama checkpoint such as SmolLM2-135M. This note grows with milestone M5; it starts with the conversion of the stored 16-bit floats to FP32.

## What it computes

Nothing trained, so the forward, backward and shapes sections of the [template](README.md) do not apply. Hugging Face stores weights in a `model.safetensors` file, mostly as BF16 (SmolLM2-135M is all BF16), sometimes as F16. NeuralTitan computes in FP32 only, so the loader widens every value to FP32. The widening is exact: each 16-bit value becomes the FP32 with the same real value, and no rounding happens anywhere.

## BF16 and F16 to FP32

### Bit layouts

A binary float is a sign bit `s`, an exponent field `e` of `E` bits stored with a bias `b = 2^(E−1) − 1`, and a mantissa field `m` of `M` bits. For a normal number (`0 < e < 2^E − 1`) the value is

```text
value = (−1)^s · (1 + m / 2^M) · 2^(e − b)
```

The three formats:

| Format | Sign | Exponent bits `E` | Bias `b` | Mantissa bits `M` | Bit positions |
| --- | --- | --- | --- | --- | --- |
| FP32 | 1 | 8 | 127 | 23 | `s` 31, `e` 30–23, `m` 22–0 |
| BF16 | 1 | 8 | 127 | 7 | `s` 15, `e` 14–7, `m` 6–0 |
| F16 | 1 | 5 | 15 | 10 | `s` 15, `e` 14–10, `m` 9–0 |

The two extreme exponent fields are special in every format:

- `e = 0` with `m = 0` is a signed zero, `+0` or `−0`.
- `e = 0` with `m ≠ 0` is a **subnormal**: `value = (−1)^s · (m / 2^M) · 2^(1 − b)`. There is no implicit leading 1, which lets values fade gradually towards zero instead of jumping from the smallest normal number to 0.
- `e = 2^E − 1` with `m = 0` is `±∞`; with `m ≠ 0` it is a NaN.

### BF16: a shift

BF16 is the upper half of FP32: the same sign bit, the same 8-bit exponent with the same bias, and the first 7 of FP32's 23 mantissa bits. So the FP32 bit pattern is the BF16 pattern followed by 16 zero bits:

```text
fp32_bits = bf16_bits << 16
```

Every case maps onto itself: the exponent field is unchanged, and the mantissa `m / 2^7` becomes `(m · 2^16) / 2^23`, the same fraction. Zeros, subnormals, infinities and NaNs keep their meaning, because those cases are decided by the exponent field alone, which does not move. For example `0x3F80` becomes `0x3F800000 = 1`, `0xC040` becomes `0xC0400000 = −3`, and the smallest BF16 subnormal `0x0001` becomes `0x00010000`, the FP32 subnormal `(2^16 / 2^23) · 2^-126 = 2^-133`, which is exactly its BF16 value `(1 / 2^7) · 2^-126`.

### F16: rebias and widen

F16 has fewer exponent bits, so the exponent field has to be rebiased. The mantissa is widened by 13 bits (`23 − 10`), which appends zeros and so cannot lose anything. Every F16 value, subnormals included, lies inside FP32's normal range (the smallest F16 subnormal is `2^-24`, far above FP32's smallest normal `2^-126`), so every F16 value is exactly representable. The exponent field picks the case:

- **Normal** (`0 < e < 31`): `2^(e − 15) = 2^((e + 112) − 127)`, so the FP32 exponent field is `e + 112`:

  ```text
  fp32_bits = (s << 31) | ((e + 112) << 23) | (m << 13)
  ```

- **Zero** (`e = 0`, `m = 0`): only the sign bit, `s << 31`, so `0x8000` becomes `−0`.
- **Subnormal** (`e = 0`, `m ≠ 0`): the value `m · 2^-24` is a normal number in FP32 and must be renormalized. Shift `m` left until its leading 1 reaches bit 10, the position of the implicit 1, counting `k` shifts. Then `m · 2^-24 = (1 + m' / 2^10) · 2^(−14 − k)`, where `m'` is the shifted mantissa without that bit, and the FP32 exponent field is `127 − 14 − k = 113 − k`:

  ```text
  fp32_bits = (s << 31) | ((113 − k) << 23) | (m' << 13)
  ```

  For `0x0001`, `m = 1` needs `k = 10` shifts, giving `2^-24`. For `0x03FF`, `m = 1023` needs `k = 1`, giving `(1 + 1022/1024) · 2^-15 = 1023 · 2^-24`.
- **Infinity and NaN** (`e = 31`): the FP32 exponent field is all ones, `0xFF`, and the mantissa moves to the top of the 23-bit field, so `m = 0` stays `±∞` and a NaN keeps its payload:

  ```text
  fp32_bits = (s << 31) | (0xFF << 23) | (m << 13)
  ```

  `0x7C00` becomes `+∞`; `0x7E00`, the quiet NaN, becomes `0x7FC00000`.

### Why exactness matters

The golden fixtures are produced by Hugging Face in FP32 from the same BF16 weights. PyTorch's `bfloat16 → float32` cast is the same 16-bit shift, so both sides start from bit-identical FP32 weights, and every difference left in the logits comes from the order of the floating-point operations, not from the loading.

### Mapping to code

| Formula | Code |
| --- | --- |
| `fp32_bits = bf16_bits << 16` | `io::bf16_to_f32` in `src/io/safetensors.cpp` |
| F16 normal, zero, subnormal, infinity and NaN cases | `io::f16_to_f32` in `src/io/safetensors.cpp` |
| The conversion of a whole tensor | `io::SafetensorsFile::read_f32`, which reads one tensor's byte range and converts it element by element |
| The bounds and contiguity checks of the file format | `io::SafetensorsFile::open`, whose layout and checks are described in `src/io/safetensors.hpp` |
