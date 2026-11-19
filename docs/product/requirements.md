# Requirements

## Problem

Modern LLMs are usually learned through high-level frameworks (PyTorch, Hugging Face) that hide the mechanics: how gradients flow through attention, what a GPU kernel actually does, how a tokenizer turns bytes into ids. The author wants a concrete, end-to-end understanding of how a modern decoder-only LLM is built, trained and run. They want to get it by reviewing a complete implementation written from first principles, where every equation is visible in code and documented.

## Users

- **The author (primary).** A senior developer who learns by reviewing each part, questioning it, and seeing how the parts combine. Runs everything locally on a Windows laptop with an RTX 3070 (8 GB).
- **Public readers (secondary).** Developers who find the repository on GitHub and want a readable reference for how an LLM works without framework abstractions.

## Goals

- G1. A composable library of LLM building blocks from which different decoder-only model shapes can be assembled through configuration.
- G2. A training mode that trains models from scratch on a text corpus.
- G3. An inference mode that generates text from a prompt.
- G4. Correctness that is demonstrable, not assumed: every operation and gradient verified, and a real pretrained model reproduced.
- G5. Understanding: every module comes with documentation of its math and how it maps to the code.

## Non-goals

- Mixed precision (FP16/BF16 compute) and quantization.
- Multi-GPU or distributed training.
- Mixture-of-experts, encoder or encoder-decoder models.
- Flash-attention-class kernel optimization.
- Python bindings, a network server, a chat UI.
- Performance parity with PyTorch or cuBLAS.

## Functional requirements

Tensor and operations

- FR-1. A tensor type (FP32, row-major, up to 4 dimensions) that lives on the CPU or the GPU, with explicit copies between devices. (G1)
- FR-2. Every operation has a CPU reference implementation and a CUDA implementation behind one device-dispatching interface. (G1, G4)
- FR-3. Operations cover:
  - matmul and batched matmul;
  - elementwise ops (add, mul, scale, SiLU, GELU);
  - reductions;
  - row softmax, including a causal variant;
  - RMSNorm and LayerNorm;
  - RoPE;
  - embedding gather and its scatter-add gradient;
  - fused cross-entropy;
  - the AdamW update;
  - the backward counterpart of each. (G1, G2)

Modules and models

- FR-4. A module interface with forward, hand-written backward, and parameter enumeration. Gradients accumulate across calls. (G1, G5)
- FR-5. Modules: Embedding, Linear, RMSNorm, LayerNorm, grouped-query attention with RoPE and a KV cache, SwiGLU MLP, GELU MLP, TransformerBlock, LM head (tied or untied), cross-entropy loss. (G1)
- FR-6. A model is assembled from a JSON config. Each transformer block has its own:
  - head count;
  - KV-head count;
  - MLP type and hidden size;
  - norm type;
  - position-encoding type (`rope` or `none`). (G1)
- FR-7. A `llama-style` preset ships with the library. (G1)

Tokenizers

- FR-8. A character-level tokenizer built from a corpus. (G2)
- FR-9. A byte-level BPE tokenizer that can:
  - train merges on a corpus;
  - encode and decode;
  - load a Hugging Face `tokenizer.json`, including the GPT-2/SmolLM2 pre-tokenization split and special tokens. (G2, G4)

Training

- FR-10. A data pipeline that tokenizes a corpus once into a binary token file and yields `B × T` training windows with next-token targets. (G2)
- FR-11. A trainer:
  - runs forward, loss, backward, gradient clipping and the AdamW step;
  - schedules the learning rate with linear warmup and cosine decay;
  - supports gradient accumulation;
  - reports validation loss periodically;
  - logs loss, tokens/s and learning rate. (G2)
- FR-12. Checkpoints save and restore model weights and optimizer state, so a training run can resume. (G2)

Inference

- FR-13. Generation:
  - prefills the prompt into the KV cache;
  - decodes token by token, streaming text to stdout;
  - samples greedily or with temperature, top-k or top-p, with a seed. (G3)
- FR-14. A safetensors loader maps Hugging Face Llama tensor names onto the model's parameters, converting BF16/FP16 to FP32. (G4)

Interface

- FR-15. A `titan` command-line program with `tokenizer train`, `train` and `generate` subcommands, and `--device cpu|cuda` on each. (G2, G3)

Documentation

- FR-16. Each module and op family has a math note in `docs/math/` covering:
  - what it computes;
  - the forward equations;
  - the backward derivation;
  - tensor shapes;
  - the mapping to the code. (G5)

## Non-functional requirements

- NFR-1 Performance:
  - The M1 CPU run (tiny Llama-style model on TinyShakespeare) completes in under 30 minutes.
  - The M6 GPU run (10–30M parameter model on TinyStories) completes in about 8 hours or less, reaching a validation loss of about 1.5 or below.
- NFR-2 Resource limits: training and inference fit in 8 GB of VRAM. SmolLM2-135M inference runs in FP32 on that GPU.
- NFR-3 Correctness:
  - Every CUDA op matches its CPU reference on random inputs within test tolerances, including shapes that are not multiples of tile sizes.
  - Every module's backward passes a finite-difference gradient check.
  - A tiny model can overfit a single batch to near-zero loss.
- NFR-4 Reproducibility: training, sampling and initialization are deterministic for a given seed on a given device. Bit-exact equality between CPU and CUDA is not required.
- NFR-5 Security: weight files, tokenizer files, checkpoints and configs are untrusted input. All parsers (JSON, safetensors, checkpoint) bounds-check every length and offset and fail with a clear error, never with out-of-bounds memory access. The project handles no credentials or personal data.
- NFR-6 Readability: the code favours explicit, commented math over cleverness. Optimized kernels keep their naive predecessor alongside them for comparison.
- NFR-7 Dependencies: only the C++ standard library and the CUDA runtime, plus cuBLAS as the default GEMM (ADR 0007). The project's own naive and tiled GEMM kernels are kept, tested and selectable for learning. Development-only scripts under `tools/` (for golden fixtures and Unicode tables) never enter the build.

## Constraints

- Windows 11, CMake, MSVC host toolset 14.38, CUDA 12.6, C++20, `sm_86` GPU target.
- Code stays portable (platform specifics isolated) but only Windows is tested.
- GitHub-hosted CI has no GPU: CI builds everything and runs CPU tests; CUDA tests run locally.
- Solo project; the code is written by the builder and reviewed by the author, in small single-purpose changes.

## Success metrics

| Metric | Baseline | Target |
| --- | --- | --- |
| CUDA ops with a passing CPU-parity test | 0 | 100% |
| Modules with a passing gradient check | 0 | 100% |
| M1: TinyShakespeare CPU run | — | < 30 min; samples recognisably Shakespeare-like |
| M6: TinyStories GPU run | — | ≤ ~8 h; validation loss ≤ ~1.5; coherent short stories |
| SmolLM2-135M logits vs Hugging Face FP32 fixtures | — | max \|Δ\| ≤ 1e-3 on every fixture prompt |
| SmolLM2-135M greedy generation | — | first 32 tokens identical (divergence only at near-ties) |
| BPE encoding vs SmolLM2 reference token ids | — | identical on all fixture strings |
| Modules and op families with a math note | 0 | 100% |

## Open questions

None.
