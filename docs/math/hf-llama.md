# Hugging Face Llama weights

How NeuralTitan reads a Hugging Face Llama checkpoint such as SmolLM2-135M. This note grows with milestone M5: the conversion of the stored 16-bit floats to FP32, then the mapping of a Hugging Face Llama onto our model.

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

## Mapping a Hugging Face Llama onto our model

### The config

A Hugging Face `LlamaForCausalLM` is our decoder-only model with one block kind repeated `num_hidden_layers` times: pre-norm RMSNorm, grouped-query attention with half-split RoPE, a SwiGLU MLP, no biases anywhere, then a final RMSNorm and the head. So `config.json` maps onto a `ModelConfig` field by field:

| `config.json` | `ModelConfig` | SmolLM2-135M |
| --- | --- | --- |
| `vocab_size` | `vocab_size` | 49152 |
| `hidden_size` | `dim` | 576 |
| `max_position_embeddings` | `context_length` | 8192 |
| `rope_theta`, or `rope_parameters.rope_theta` | `rope_theta` | 100000 |
| `rms_norm_eps` | `norm_eps` | 1e-5 |
| `tie_word_embeddings` (default false) | `tied_embeddings` | true |
| `num_hidden_layers` | the number of blocks | 30 |
| `num_attention_heads` | `n_heads` of every block | 9 |
| `num_key_value_heads` (default `num_attention_heads`) | `n_kv_heads` of every block | 3 |
| `intermediate_size` | `mlp_hidden` of every block | 1536 |
| (fixed) | `mlp` "swiglu", `norm` "rmsnorm", `position` "rope", `final_norm` "rmsnorm" | |

The head dimension is `hidden_size / num_attention_heads` (576 / 9 = 64 for SmolLM2), so the `k_proj` and `v_proj` output size is `num_key_value_heads · 64 = 192`. RoPE rotates the pairs `(i, i + 32)` of each head, which is Hugging Face's `rotate_half` layout and the one of [rope.md](rope.md).

Every other key either changes the computation or does not. The keys that would change it are accepted only at the value our model computes: `hidden_act` "silu" (SwiGLU's gate is SiLU), `attention_bias` and `mlp_bias` false (our projections have no bias), `rope_scaling` null and `rope_parameters.rope_type` "default" (no rescaled frequencies), `rope_interleaved` false (pairs `(i, i + hd/2)`, not `(2i, 2i + 1)`), `pretraining_tp` 1 (no sliced MLP matmuls) and `head_dim` equal to `hidden_size / num_attention_heads`. Informational keys (`dtype`, `initializer_range`, `use_cache`, ...) are ignored, and any key outside both lists is rejected, so a config of a different architecture cannot load silently as a wrong model. transformers 4.x writes `rope_theta` and `torch_dtype`, 5.x writes `rope_parameters` and `dtype`; both forms load.

### The tensors

Both libraries store a linear layer's weight as `(out, in)` and compute `y = x · Wᵀ`, so every tensor is copied as it is, with no transpose. Only the names differ:

| Hugging Face | Ours | Shape |
| --- | --- | --- |
| `model.embed_tokens.weight` | `embed_tokens.weight` | `(vocab, dim)` |
| `model.layers.N.input_layernorm.weight` | `blocks.N.norm1.weight` | `(dim)` |
| `model.layers.N.self_attn.q_proj.weight` | `blocks.N.attn.q_proj.weight` | `(dim, dim)` |
| `model.layers.N.self_attn.k_proj.weight` | `blocks.N.attn.k_proj.weight` | `(n_kv_heads · head_dim, dim)` |
| `model.layers.N.self_attn.v_proj.weight` | `blocks.N.attn.v_proj.weight` | `(n_kv_heads · head_dim, dim)` |
| `model.layers.N.self_attn.o_proj.weight` | `blocks.N.attn.o_proj.weight` | `(dim, dim)` |
| `model.layers.N.post_attention_layernorm.weight` | `blocks.N.norm2.weight` | `(dim)` |
| `model.layers.N.mlp.gate_proj.weight` | `blocks.N.mlp.gate_proj.weight` | `(hidden, dim)` |
| `model.layers.N.mlp.up_proj.weight` | `blocks.N.mlp.up_proj.weight` | `(hidden, dim)` |
| `model.layers.N.mlp.down_proj.weight` | `blocks.N.mlp.down_proj.weight` | `(dim, hidden)` |
| `model.norm.weight` | `final_norm.weight` | `(dim)` |
| `lm_head.weight` (untied only) | `lm_head.weight` | `(vocab, dim)` |

With tied embeddings the head computes `logits = h · Eᵀ` with the embedding table `E` itself, so the file holds no `lm_head.weight` and one is rejected; without them it must be present. The q and k projections need no permutation either: the permutation some conversion scripts apply exists only to turn Meta's interleaved RoPE into `rotate_half`, and the Hugging Face weights are already in the `rotate_half` layout.

For SmolLM2 the counts are `49152 · 576 = 28,311,552` for the tied embedding, `2 · 576 + 2 · 576² + 2 · 192 · 576 + 3 · 1536 · 576 = 3,540,096` per layer, and 576 for the final norm: `28,311,552 + 30 · 3,540,096 + 576 = 134,515,008` parameters.

### Loading order

`load_hf_model` treats every file as untrusted and changes nothing until everything is checked:

1. `config.json` is parsed and validated first, before any other file is opened.
2. `tokenizer.json` must have `vocab_size` tokens and `eos_token_id` as its end-of-text token.
3. From the safetensors header alone, every tensor name must map to a distinct parameter, every parameter must have a tensor, and the element counts must add up to the config's parameter count. The header is bounded by the file size, so a config that claims a huge model fails here instead of in an allocation.
4. The model is built, and each tensor's shape must equal its parameter's.
5. The tensors are read one at a time and widened to FP32 into the parameters.

### Mapping the loader to code

| Step | Code |
| --- | --- |
| `config.json` to `ModelConfig`, with every key's check | `runtime::parse_hf_llama_config` in `src/runtime/hf_model.cpp` |
| The name table | `runtime::hf_to_titan_name` and `kLayerTensors` in `src/runtime/hf_model.cpp` |
| The loading order and the checks of each step | `runtime::load_hf_model` in `src/runtime/hf_model.cpp` |
| The tiny test models | `tools/make_llama_fixtures.py --tiny`, which writes `tests/fixtures/tiny_llama/` |
