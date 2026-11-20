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
