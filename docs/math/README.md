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
