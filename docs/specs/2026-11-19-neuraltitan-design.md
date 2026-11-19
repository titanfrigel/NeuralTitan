# NeuralTitan — Design

Date: 2026-11-19
Status: Draft, awaiting approval

## 1. Intent

NeuralTitan is a from-scratch LLM library in C++ and CUDA. Its purpose is **learning**: the author wants to understand, deeply and concretely, how a modern large language model works — tokenization, embeddings, attention, MLPs, normalization, position encoding, the backward pass, optimization, and inference — by having every piece built from first principles and reviewing how each part works and how the parts combine.

It is a *library*, not a single model: building blocks (tokenizers, modules, optimizers, samplers) are composed from a configuration into a model, so different LLM shapes can be built and compared. A `titan` executable drives the two modes: **training** and **inference** (generation).

Readability and understanding beat performance everywhere they conflict.

## 2. Goals, non-goals, anti-goals

### Goals
- Implement a modern decoder-only transformer in the style of today's open models (Llama 3, Qwen, Mistral, SmolLM): RMSNorm (pre-norm), RoPE, SwiGLU MLP, grouped-query attention (GQA), tied or untied embeddings, KV cache for generation.
- Every model is described by a config; each transformer block is instantiated with its own attention and MLP settings (heterogeneous blocks are allowed).
- Every operation exists twice: a plain, readable **CPU reference** and a **CUDA** implementation, verified against each other.
- Every module has a **hand-written backward pass**, verified by finite-difference gradient checks.
- Tokenizers: character-level first, then byte-level BPE (train our own, and load a Hugging Face `tokenizer.json`).
- End-to-end training on a small corpus (TinyShakespeare, then TinyStories) producing coherent text.
- **Final correctness target:** load the real **SmolLM2-135M** weights (Llama architecture) and reproduce its reference logits and its greedy generations.
- Each module's math (forward and backward derivation) is documented alongside the code — the documentation is part of the learning output.

### Non-goals (v1)
- Mixed precision (FP16/BF16 compute), quantization.
- Multi-GPU, distributed training.
- Mixture-of-experts, encoder or encoder-decoder models.
- Flash-attention-class kernel optimization (a later stage, not v1).
- Python bindings, a server, a chat UI.
- Competitive performance with PyTorch/cuBLAS.

### Anti-goals (a failure even if it works)
- Any third-party library in the build (no cuDNN, CUTLASS, Eigen, JSON libraries, test frameworks). The only allowed dependencies are the C++ standard library and the CUDA runtime. The one exception is cuBLAS, as an optional, off-by-default GEMM backend (ADR 0006); the project's own GEMM kernels stay the default.
- "Magic" that hides the math: a generic autograd engine in v1, or code transcribed from llm.c / PyTorch without the author being able to explain it.
- CUDA kernels that are not checked against the CPU reference.

## 3. Constraints

- **Platform:** Windows 11, MSVC host toolset **14.38** (installed side by side in Visual Studio 2026; the default 14.50 is not supported by CUDA 12.6), **CUDA 12.6**, CMake.
- **GPU:** RTX 3070 Laptop, 8 GB VRAM, compute capability 8.6 (`sm_86`).
- **Language:** C++20 (host), CUDA C++ (device). Code is kept portable (no Windows-only APIs outside a thin platform layer) so a Linux build is possible later, but only Windows is tested.
- **Precision:** FP32 for parameters, activations, gradients and optimizer state.
- **Memory budget:** AdamW training costs ~16 bytes/parameter before activations; 8 GB supports training models up to roughly 50–100M parameters at modest batch × context, and inference of SmolLM2-135M (~540 MB in FP32) comfortably.

## 4. Approaches considered

| Approach | Summary | Trade-off | Decision |
|---|---|---|---|
| **A. Module objects, hand-written backward** | `Tensor` + `Module { forward, backward, parameters }`; composite modules call children's backward in reverse order; model assembled from config. | All gradient math visible; new architectures need some glue code in composite modules. | **Chosen.** |
| B. Compute graph / autograd | Config builds a graph of primitive ops; engine derives backward order and plans memory. | New architectures nearly free, but ~2× infrastructure before the first token, and the math hides behind the engine. | Rejected for v1; a possible later learning stage built on top of A. |
| C. Monolithic llm.c-style struct | One struct of all weights/activations, one function per layer. | Fastest to a working model, but not a library and cannot express different model forms. | Rejected. |

## 5. Architecture

Layers, from bottom to top. Each layer depends only on the ones below it.

```
titan (CLI: train / generate / tokenizer)
  └─ runtime: Trainer, Generator, Sampler, DataLoader, Checkpoint
      └─ model: Model, ModelConfig, presets (llama-style, gpt2-style), weight loaders
          └─ nn: Module, Parameter, Embedding, RMSNorm, LayerNorm, Linear,
                 Attention (GQA + RoPE + KV cache), SwiGLU, GeluMLP,
                 TransformerBlock, LMHead, CrossEntropyLoss
              └─ optim: AdamW, LR schedules, gradient clipping
              └─ ops: device-dispatching operations (matmul, softmax, ...)
                  ├─ cpu/  reference implementations
                  └─ cuda/ kernels
                      └─ core: Tensor, Storage, Device, allocator, RNG, errors
tokenizer: CharTokenizer, BpeTokenizer (train, encode, decode, load tokenizer.json)
io: JSON parser, safetensors reader, binary checkpoint format
```

### 5.1 core
- `Device { CPU, CUDA }`.
- `Tensor`: FP32, row-major contiguous, up to 4 dimensions, owns (or shares) a `Storage` on one device. Explicit `to(Device)` copies; no implicit transfers. Views over contiguous sub-ranges (e.g. a row slice) share storage.
- A simple caching allocator for CUDA memory so training steps do not call `cudaMalloc` repeatedly.
- Deterministic RNG (seeded) for initialization, dropout if added, sampling, and data shuffling.
- Error handling: a `TITAN_CHECK` macro and CUDA error checking that throws with file/line.

### 5.2 ops
Each op has one signature in `ops/` that dispatches on the tensors' device to `cpu::` or `cuda::`. Initial set:
- matmul (with transposes), batched matmul
- elementwise: add, mul, scale, SiLU, GELU, residual add
- reductions: sum, row max
- softmax (row-wise, causal-masked variant)
- RMSNorm, LayerNorm
- RoPE (apply / inverse for backward)
- embedding gather, embedding gradient scatter-add
- cross-entropy (fused softmax + NLL, with backward)
- AdamW update
- backward counterparts for each of the above

CUDA matmul is written in stages, each kept and benchmarked: naive → shared-memory tiling → register tiling. Attention in v1 materializes the score matrix (no flash attention).

### 5.3 nn
```cpp
struct Parameter { std::string name; Tensor value; Tensor grad; };

class Module {
public:
  virtual Tensor forward(const Tensor& x, ForwardContext& ctx) = 0;
  virtual Tensor backward(const Tensor& dy) = 0;   // returns dx, accumulates param grads
  virtual void parameters(std::vector<Parameter*>& out) = 0;
  virtual ~Module() = default;
};
```
- In **training mode** `forward` caches what its `backward` needs; in **inference mode** it caches nothing and attention uses the KV cache carried in `ForwardContext` (positions, cache slots).
- `TransformerBlock` = `x + Attn(Norm(x))`, then `x + MLP(Norm(x))`; its MLP and attention are chosen from its own block config.
- Gradients accumulate (`+=`) so gradient accumulation across micro-batches works; `zero_grad` clears them.
- Initialization schemes are per-module and documented.

### 5.4 model
- `ModelConfig`: vocab size, model dim, context length, RoPE theta, norm epsilon, tied embeddings flag, and a list of block configs (`n_heads`, `n_kv_heads`, MLP type and hidden size, norm type, position encoding type).
- Config file format: JSON, parsed by our own parser. Presets: `llama-style` (default) and `gpt2-style` (LayerNorm, learned positional embedding, GELU MLP) — the latter optional after v1's core milestones.
- Weight loaders: our own checkpoint format (save/load including optimizer state for resume), and a **safetensors** reader that maps Hugging Face Llama tensor names onto our parameters, converting BF16/FP16 to FP32 on load.

### 5.5 tokenizer
- `CharTokenizer`: vocabulary built from the corpus.
- `BpeTokenizer`: byte-level BPE. Train merges on a corpus; encode/decode; load Hugging Face `tokenizer.json` (vocab + merges + the pre-tokenization regex behaviour needed for SmolLM2, hand-implemented since no regex library is used beyond what is strictly necessary).
- Special tokens (BOS/EOS) supported.

### 5.6 runtime
- `DataLoader`: corpus tokenized once to a binary file of token ids; yields random (or sequential) `B × T` windows with next-token targets.
- `Trainer`: loop of forward → loss → backward → clip → AdamW step → schedule; gradient accumulation; periodic validation loss; periodic checkpoint; logs loss, tokens/s, LR.
- LR schedule: linear warmup + cosine decay.
- `Generator`: prompt → encode → prefill (full prompt through the model, filling the KV cache) → decode loop one token at a time → decode text, streaming to stdout.
- `Sampler`: greedy, temperature, top-k, top-p; seeded.

### 5.7 CLI (`titan`)
```
titan tokenizer train --corpus data/shakespeare.txt --vocab 4096 --out tok.json
titan train    --config configs/tiny-llama.json --data data/tinystories --out runs/tiny
titan generate --checkpoint runs/tiny/latest.ckpt --prompt "Once upon a time" --temp 0.8 --top-p 0.9
titan generate --hf-model models/SmolLM2-135M --prompt "..."
```
`--device cpu|cuda` on all commands; CUDA is the default when available.

## 6. Verification strategy

- **Own test harness** (a small header: test registration, assertions with tolerances, a runner executable). Tests run through CTest.
- **Op parity:** every CUDA op is tested against the CPU reference on random inputs, including awkward shapes (non-multiples of tile sizes).
- **Gradient checks:** every module's backward is checked against central finite differences on the CPU. The CPU reference is written so these checks are numerically meaningful (builder's choice: templated on float/double, or careful epsilon/tolerance in FP32).
- **Training sanity:** a tiny model must overfit a single batch to near-zero loss.
- **Golden reference for SmolLM2-135M:** a one-off script under `tools/` (Python + Hugging Face `transformers`, used only on the author's machine, never part of the build) dumps tokenizer output and logits for a few prompts into fixture files; the C++ tests compare against them. *This is a development tool, not a dependency of the library.*

## 7. Milestones

Each milestone ends with something runnable and tested.

| # | Milestone | Done when |
|---|---|---|
| M0 | Skeleton | CMake builds a library, the `titan` CLI and the test runner with MSVC 14.38 + CUDA 12.6 for `sm_86`; `Tensor` on CPU and CUDA with copies; test harness runs. |
| M1 | CPU model + training | All CPU ops, all modules with gradient checks, AdamW, char tokenizer; a tiny Llama-style model trains on TinyShakespeare on CPU and its loss falls; text samples look Shakespeare-ish. |
| M2 | CUDA backend | Every op has a CUDA kernel passing parity tests; the M1 run trains on the GPU, much faster; staged matmul kernels benchmarked. |
| M3 | Inference | KV cache, prefill/decode, samplers, checkpoint save/load/resume; `titan generate` streams text from an M2 checkpoint. |
| M4 | BPE tokenizer | Train BPE on a corpus; load `tokenizer.json`; encoding matches SmolLM2's reference token ids on fixture strings. |
| M5 | SmolLM2-135M | safetensors loader; logits match the golden fixtures within tolerance; greedy generations match. |
| M6 | Train a real small model | A ~10–30M parameter Llama-style model trained from scratch on TinyStories with our BPE produces coherent short stories. |

### Later stages (out of v1, listed so v1 does not block them)
GPT-2 preset and GPT-2 124M loading; kernel optimization and flash attention; mixed precision; a compute-graph autograd built on top of the modules (approach B); mixture-of-experts.

## 8. Decided by the author
- Dependencies: C++ standard library and CUDA runtime only, except optional cuBLAS for GEMM (ADR 0006, decided after the spec's first approval).
- Gradients: hand-written backward per module, verified by gradient checks.
- CPU reference first, CUDA verified against it.
- Architecture family: modern open-model style (Llama-like decoder).
- Success: train small models to coherent text, then reproduce a real pretrained model exactly.
- Build: CMake, portable code, tested on Windows only.
- Library structure: approach A.
- **Who writes the code:** the builder writes all of it. The author learns by reviewing and understanding each part and how the parts combine. Consequences:
  - The work is delivered in small, single-purpose, reviewable steps: one module or op family per change. Each step comes with its tests and its math note.
  - The math notes in `docs/math/` are a first-class deliverable, not an afterthought. Each one covers:
    - what the module computes;
    - the forward equations;
    - the full backward derivation;
    - shapes;
    - how it maps onto the code.
  - Code favours explicit, commented math over cleverness. Optimized kernels keep their naive predecessor alongside them for comparison.
- **Compute budget:**
  - M1 (CPU, TinyShakespeare) completes in under 30 minutes.
  - M6 (GPU, TinyStories) completes overnight (≤ ~8 hours), reaching a validation loss of about 1.5 or lower.
  - These targets decide how far matmul optimization must go: register tiling is required if the naive and shared-memory kernels cannot meet the M6 budget.

## 9. Proposed by the designer (correctable)
- SmolLM2-135M as the pretrained target, because it is a true Llama-architecture model that fits the 8 GB GPU in FP32 and uses byte-level BPE.
- Tokenizer path: character-level, then byte-level BPE (author had no preference).
- FP32 only in v1 (author had no preference); mixed precision deferred because it adds numerical-stability issues unrelated to learning the fundamentals.
- C++20; JSON as the config format; own JSON parser and own binary checkpoint format.
- A Python + `transformers` script under `tools/` to produce golden fixtures, outside the build.
- Attention materializes the score matrix in v1.
- **CPU threading:** the CPU reference ops are single-threaded and written for clarity (they back the gradient checks). Hot CPU ops (matmul first) may add a `std::thread`-based parallel-for. OpenMP is excluded: it ships with MSVC but is not the standard library.
- **Unicode for the BPE pre-tokenizer:** `std::regex` cannot express `\p{L}` / `\p{N}`, so the GPT-2/SmolLM2 split pattern is a hand-written scanner. The Unicode category tables it needs are generated once from `UnicodeData.txt` by a `tools/` script and committed as C++ source.
- **Position encoding per block:** `rope` or `none` (NoPE) in v1. RoPE uses the half-split layout (`rotate_half`) that Hugging Face Llama checkpoints use, so their Q/K weights need no permutation on load. No RoPE scaling in v1. Learned absolute positions belong to the later gpt2-style preset and are model-level, not per block.
- **SmolLM2 match criteria:** golden fixtures are produced by Hugging Face in **FP32**, with the BF16 weights upcast exactly as our loader does. Pass conditions:
  - logits for each fixture prompt match within `max |Δ| ≤ 1e-3` (relative to logit scale);
  - greedy generation matches the first 32 tokens;
  - a divergence is accepted only at a near-tie, where the top-2 logit gap is below the tolerance.
  
  Bit-exact determinism across CPU and CUDA is not required.

## 10. Left to the builder
- Exact file layout inside each layer, naming, class boundaries beyond the ones above.
- Allocator design, kernel launch configurations, tile sizes.
- Test tolerances, gradient-check precision strategy.
- Checkpoint binary layout.
- Logging format and CLI flag details beyond the examples.
