# Hugging Face Llama weights

How NeuralTitan reads a Hugging Face Llama checkpoint such as SmolLM2-135M. It covers the conversion of the stored 16-bit floats to FP32, the mapping of a Hugging Face Llama onto our model (its config, its tensors and the load order), and the [Tolerance](#tolerance) the parity tests hold our logits to.

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

## Tolerance

The pass condition of M5 is that our model computes what Hugging Face computes. Both sides run in FP32 from bit-identical weights (see [Why exactness matters](#why-exactness-matters)), but they cannot be expected to agree bit for bit, so the comparison needs a tolerance and a rule for greedy generation.

### What is compared

The reference is Hugging Face transformers in FP32 on the CPU with eager attention, run by `tools/make_llama_fixtures.py --golden`. For every fixture prompt:

- **Logits.** For the two tiny models, the full logits `(T, 512)` at every prompt position. For SmolLM2-135M, the full last-position logits (49152 values) and the logits of 1024 seeded vocabulary ids at every prompt position (`T · 1024` values); storing all `T · 49152` would make the fixture several megabytes. Ours come from one inference forward over the whole prompt. All the values of a prompt form one comparison, with one scale.
- **Greedy tokens.** 16 tokens for the tiny models and 32 for SmolLM2, from `generate(do_sample=False)` on the reference side and from our KV-cached `Generator` at temperature 0 on ours, both stopping at end-of-text. A run that stops there has the end-of-text token as its last step on both sides.

### The logit tolerance is relative

A prompt passes when

```text
max |ours − ref| ≤ tol · scale,   scale = max(1, max |ref|),   tol = 1e-3
```

over every compared value of the prompt. The bound grows with the logits because FP32 rounding errors do: each operation rounds its result to 24 significant bits, a relative error of at most `2^-24 ≈ 6e-8` of the value, so the absolute error of a logit near 30 is about 30 times that of a logit near 1. A fixed absolute bound would be loose for small models and too tight for large ones. The `max(1, ·)` floor keeps a prompt whose logits are all small from being held to a bound below `tol` itself, where the absolute rounding noise of the intermediate activations (which are not small) would dominate.

### Why FP32 sums differ between implementations

Floating-point addition is not associative: `(a + b) + c` and `a + (b + c)` round differently. Every matrix product, RMSNorm and softmax is a long sum (576 to 49152 terms here), and each implementation orders it its own way:

- PyTorch's CPU matmul (oneDNN or MKL) splits the sum into SIMD lanes and cache blocks, and fuses multiply and add.
- Our CPU matmul sums blocks of 64 terms in FP32 and adds the blocks in FP64 (`src/ops/cpu/matmul.cpp`), and our reductions accumulate in their own order.
- cuBLAS picks a kernel, and with it a summation tree, by the matrix shape; our CUDA reductions sum per thread and then across a block.
- `exp`, `rsqrt` and `sin`/`cos` of the RoPE angles come from different math libraries, each correct to about one unit in the last place but not identical.

Each difference is a few units in the last place of one value, and 30 layers pass them on through the residual stream, so the logits differ by about `1e-6` relative to their scale. That is three orders of magnitude below the tolerance, which is set to catch a real error: a wrong mapping or a wrong constant changes the logits by a large fraction of their scale (below).

### The near-tie rule

Greedy decoding takes the argmax. When the reference's two best logits at a step are closer than the logit tolerance, an error within tolerance can swap them, and from that step on the two runs continue different texts. So a greedy run passes when every token matches, or when the first mismatch is at a step whose reference top-2 gap (largest minus second-largest raw FP32 logit) is strictly below `tol = 1e-3`; the comparison of that prompt stops there. A mismatch at any step with a wider gap fails.

The gap is compared with the absolute `tol`, not `tol · scale`, so the rule accepts fewer divergences than the logit bound alone would justify. With the measured errors (about `1e-4` absolute) every step with a gap above `1e-3` is decided the same way on both sides; the smallest reference gaps of the fixture runs are `3.0e-3` (tiny untied, prompt 0) and `7.7e-3` (SmolLM2, prompt 3).

### Measured errors

From `HfGolden.*` in `tests/runtime/hf_golden_test.cpp` (release build, an RTX 3070 Laptop GPU), as `max |ours − ref| / scale`:

| Prompt | CPU | CUDA | Greedy, CPU and CUDA |
| --- | --- | --- | --- |
| tiny tied, `to be or not` | 8.8e-7 | 6.5e-7 | 16 / 16 |
| tiny tied, `the king` | 8.8e-7 | 1.4e-6 | 16 / 16 |
| tiny tied, `ROMEO:\n` | 9.2e-7 | 1.2e-6 | 16 / 16 |
| tiny untied, `to be or not` | 7.7e-7 | 8.1e-7 | 16 / 16 |
| tiny untied, `the king` | 8.6e-7 | 1.7e-6 | 16 / 16 |
| tiny untied, `ROMEO:\n` | 7.6e-7 | 1.1e-6 | 16 / 16 |
| SmolLM2, `Once upon a time` | 2.0e-6 | 2.8e-6 | 32 / 32 |
| SmolLM2, `The capital of France is` | 4.9e-6 | 3.0e-6 | 32 / 32 |
| SmolLM2, `def fibonacci(n):\n` | 2.8e-6 | 3.6e-6 | 32 / 32 |
| SmolLM2, `Café au lait, s'il vous plaît.` | 2.8e-6 | 4.7e-6 | 32 / 32 |

The scales are 5.3 to 6.1 for the tiny models and 21 to 33 for SmolLM2. The worst error, `4.9e-6`, is 200 times below the tolerance, and no greedy run needed the near-tie rule. The README's [M5 result](../../README.md#m5-result) adds SmolLM2's 64-token greedy output from `titan generate --hf-model` and its speed on the CPU and CUDA.

The tests do catch real mistakes. Swapping `input_layernorm` and `post_attention_layernorm` in the name map gives errors of 0.54 to 0.68 times the scale on the tiny models and 1.1 to 1.3 times on SmolLM2, and greedy divergences at step 0 or 1 for every prompt. Loading `rope_theta` as 10000 instead of 100000 gives 0.09 to 0.22 times the scale on the tiny models and 0.02 to 0.13 on SmolLM2, and greedy divergences at gaps of 0.07 to 0.8. A fresh Hugging Face model has every norm weight equal to 1, which would make the norms interchangeable and hide the swap, so the fixture script sets the tiny models' norm weights to `1 + 0.1 · N(0, 1)` before saving them.

### Mapping the comparison to code

| Rule | Code |
| --- | --- |
| `scale = max(1, max \|ref\|)`, `max \|Δ\| ≤ tol · scale` | `test::compare_logits` in `tests/support/golden.cpp` |
| The near-tie rule | `test::compare_greedy` in `tests/support/golden.cpp` |
| The comparisons per model, prompt and device | `HfGolden.*` in `tests/runtime/hf_golden_test.cpp` |
| `tol = 1e-3`, which the fixtures must record, and the prompt ids, which must be what our tokenizer makes of the prompt text | `kTolerance` and `HfGolden.PromptIdsMatchTokenizer` in `tests/runtime/hf_golden_test.cpp` |
| The reference values | `tools/make_llama_fixtures.py --golden` |
