# Math notes

One note per module or op family. Each note explains the math behind the code and maps it to the source, so you can read the two side by side.

## Template

Every note has these sections, in this order:

1. **What it computes**: the operation in words and the problem it solves.
2. **Forward**: the formulas.
3. **Backward derivation**: the gradient with respect to every input and parameter, derived step by step.
4. **Shapes**: input, output and parameter shapes.
5. **Mapping to code**: which file, class or kernel implements each formula.

Notes about infrastructure with no gradient, such as tensor memory, replace the forward and backward sections with the rules the code follows. Math is written in plain Unicode and inline code, so it reads the same in any viewer.

## Index

| Note | Covers |
| --- | --- |
| [Tensor memory](tensor-memory.md) | Row-major layout, views that share storage, host and device memory, the CUDA caching allocator. |
| [CUDA execution](cuda-execution.md) | Grids, blocks, warps and the grid-stride loop, shared memory, deterministic reductions, the default stream, and why the caching allocator is safe on one stream. |
| [Elementwise ops](elementwise.md) | Add, multiply, scale, SiLU and GELU, with their gradients. |
| [Reductions](reduce.md) | Sum, sum of squares, row maximum, sum over leading dimensions and the finiteness check. |
| [Matmul](matmul.md) | Matrix product with transposes and batching, its gradients, and the parallel CPU kernel with its deterministic accumulation order. |
| [Softmax](softmax.md) | Row softmax and causal softmax, why the mask comes before the maximum, and the backward formula from the softmax Jacobian. |
| [Normalization](normalization.md) | RMSNorm and LayerNorm, with the backward formulas derived including the projection term. |
| [RoPE](rope.md) | Rotary position embedding in the half-split layout, the relative-position property, and the backward pass as the inverse rotation. |
| [Embedding](embedding.md) | Token-id lookup as a row gather, and its backward pass as a scatter-add that sums the gradients of repeated ids. |
| [Cross-entropy](cross-entropy.md) | Fused softmax cross-entropy with log-sum-exp, and the gradient `(softmax − onehot)/N` derived from it. |
| [Linear](linear.md) | The affine map `y = x Wᵀ + b` over the last dimension, flattened to rows, and its three gradients. |
| [Feed-forward blocks](mlp.md) | SwiGLU and the GELU MLP, with the backward chain through the gate, the activation and both projections. |
| [Attention](attention.md) | Causal grouped-query attention with RoPE: the scaled dot-product, the causal mask, the head layout ops and the group-sum gradient of the shared KV heads, every shape through the layer. |
| [KV cache](kv-cache.md) | Reusing the keys and values of earlier positions during generation: why causality allows it, prefill against decode, `O(T)` scores per token, the memory formula and the GQA saving. |
| [Transformer block](transformer-block.md) | The pre-norm residual block `h = x + attn(norm1(x))`, `y = h + mlp(norm2(h))`, and how the residual gradient splits into both paths. |
| [Model](model.md) | The full decoder-only model: the forward chain with shapes, the backward order, weight tying as a sum of two gradients, the GPT-2 initialization and the parameter-count formula. |
| [AdamW](adamw.md) | AdamW with bias-corrected moments and decoupled weight decay, no decay on rank-1 parameters, global-norm gradient clipping, and the warmup + cosine learning-rate schedule. |
| [Sampling](sampling.md) | Temperature, top-k and top-p on the last position's logits, why an inverse-CDF draw with one uniform is exact, and the order of operations with its RNG contract. |
| [Byte-level BPE](bpe.md) | SmolLM2's tokenizer end to end: what a tokenizer computes; the pre-tokenizer (individual digits, then the GPT-2 split pattern); the byte-level alphabet; encode and why the merge order matches Hugging Face's; training with the lazy-heap incremental update; decode and UTF-8 streaming; the `tokenizer.json` file format. |
| [Hugging Face Llama weights](hf-llama.md) | Loading a Hugging Face Llama checkpoint: the exact BF16 and F16 to FP32 conversion (bit layouts, rebiasing, subnormals, infinities and NaN), the mapping of `config.json` and of the tensors onto our model, the load order, and the relative logit tolerance of the parity tests. |
