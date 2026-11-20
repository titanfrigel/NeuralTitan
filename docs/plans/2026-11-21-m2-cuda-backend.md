# M2 CUDA Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every op has a CUDA kernel that passes a parity test against its CPU reference. The M1 TinyShakespeare run trains on the GPU, much faster than on the CPU. The naive, tiled and cuBLAS matmul kernels are benchmarked against the GPU's roofline.

**Architecture:**

- `src/ops/cuda/` becomes the CUDA backend of `ops`: one `.hpp` (plain C++ declarations, no CUDA header) and one `.cu` per op family, mirroring `ops/cpu`. It links `CUDA::cudart` and `CUDA::cublas` privately and may include only `core` (overview §10).
- The dispatch functions in `src/ops/*.cpp` validate as before, then route on the device: every tensor on the CPU goes to `ops::cpu::`, every tensor on CUDA goes to `ops::cuda::`, and a mix throws.
- Layers above `ops` stay device-agnostic. A model is built and initialized on the CPU, then moved with `Model::to(Device)`. The trainer and the CLI take a device.
- Token ids stay host `std::span<const std::int32_t>`. The CUDA embedding and cross-entropy upload them per call.

**Tech Stack:** C++20, CUDA 12.6 for `sm_86`, cuBLAS (ADR 0007), MSVC 14.38, CMake presets, the project's own test harness. No new dependencies.

**Spec:** `docs/specs/2026-11-19-neuraltitan-design.md` (§5.2, §6, §7 M2), with `docs/product/requirements.md`, `docs/architecture/overview.md` §10, ADRs 0005 and 0007.

## Global Constraints

- **FP32 everywhere.** Parameters, activations, gradients and optimizer state are FP32. One seed drives initialization, data order and sampling.
- **Dispatch.**
  - `src/ops/<family>.cpp` keeps all validation. The `ops::cuda::` functions do not validate.
  - A mix of devices throws `ops::<name>: tensors are on different devices (cpu vs cuda)`, naming the devices in the order the arguments appear.
  - Until a family's task lands, its dispatch keeps `detail::require_cpu`. The last op task deletes `require_cpu`.
- **Kernels.**
  - Every launch is followed by `TITAN_CUDA_CHECK(cudaGetLastError())`. All work runs on the default stream.
  - Element-parallel kernels use grid-stride loops, so any element count is covered. Row kernels put one block per row on `gridDim.x` and never use `gridDim.y` for rows.
  - **Deterministic:** no floating-point `atomicAdd`. Reductions use a fixed tree order, so a repeated run on the same GPU gives bit-identical results. Overview §5 requires this.
  - Row reductions (softmax, norms, cross-entropy) accumulate in float with a fixed block tree. Whole-tensor reductions (`sum`, `sum_squares`) and the column sums that produce weight gradients (`sum_leading`, norm `dweight`/`dbias`, `embedding_backward`) accumulate in `double`.
  - Every optimized kernel keeps its naive version alongside it.
- **GEMM (ADR 0007).**
  - `enum class GemmBackend { cublas, naive, tiled }`, with `ops::set_gemm_backend(GemmBackend)` and `ops::gemm_backend()` in `src/ops/matmul.hpp`. The default is `cublas`. CPU matmuls ignore it.
  - cuBLAS runs with `CUBLAS_PEDANTIC_MATH`, so no TF32 is used. The row-major mapping is `Cᵀ = Bᵀ·Aᵀ`. Batched products use `cublasSgemmStridedBatched`.
  - The tiled kernel uses 16×16 shared-memory tiles and 16×16 thread blocks.
- **Parity tests.**
  - `tests/support/parity.hpp` provides `TITAN_REQUIRE_CUDA()`, which skips the test via `TITAN_SKIP` when `!cuda_available()`. It also provides `TITAN_EXPECT_PARITY(cpu_tensor, cuda_tensor, atol, rtol)`, which passes when `|c − g| ≤ atol + rtol·|c|` for every element and on failure reports the worst element's index and both values.
  - The default tolerances are `atol = 1e-5, rtol = 1e-4`. An op that needs looser ones states them in a comment with the reason.
  - Each family's parity test covers at least:
    - a single element;
    - a shape that is not a multiple of 16 or 32 (for example 37×129);
    - one tensor of at least 2²⁰ elements.
- **Math notes.** Each op family's existing note in `docs/math/` gains a "CUDA implementation" section: the thread mapping, the reduction order, and what differs from the CPU. `docs/math/cuda-execution.md` (new) explains grids, blocks, warps, shared memory, the default stream and the caching allocator. `docs/math/matmul.md` gets the roofline section.
- **Change rules.**
  - `CMakeLists.txt` files are edited through the shell, never with the Edit/Write tools.
  - Conventional Commits, no AI attribution, English only.
  - Branch `feat/m2-cuda-backend`, stacked on `feat/m1-cpu-training`.
  - `scripts/check.ps1` prints `check: OK` after every task. CI has no GPU, so CUDA tests skip there and must pass locally.

## Review Focus

1. **Mixed devices.** A CPU tensor passed with a CUDA tensor fails with the device-mismatch message, never a host read of a device pointer. Pinned in Task 1 (`OpsDispatch.MixedDevicesThrow`).
2. **Awkward and large shapes.** These match the CPU and need no grid dimension above its limit:
   - row counts above 65535 (`gridDim.y`'s limit);
   - sizes that are not tile multiples;
   - single elements.

   Pinned in Task 2 (`RowMaxParity.ManyRows`, 70000 rows) and Task 4 (`SoftmaxParity.ManyRows`).
3. **Tied embeddings after `Model::to(Device::CUDA)`.** The head and the embedding still share one storage; otherwise training silently differs from the CPU. Pinned in Task 9 (`Model.ToCudaKeepsTiedWeights`).
4. **Out-of-range token ids on CUDA.** They are rejected on the host before upload, with the CPU's message, because kernels do not validate. Pinned in Task 7 (`EmbeddingParity.RejectsOutOfRangeIdOnCuda`, `CrossEntropyParity.RejectsOutOfRangeTargetOnCuda`).
5. **`--device cuda` without a GPU.** It fails with `error: no CUDA device available` and exit code 1, before anything is written to `--out`. `--gemm` other than `cublas` with `--device cpu` fails with `error: --gemm needs --device cuda`. Pinned in Task 10.

## Decisions this plan makes (correctable by the author)

- **Init on the CPU, then move.** The same seed gives the same initial weights on both devices. `Rng` and `nn::init` stay CPU-only.
- **Device default.** `titan train` defaults to `cuda` when a GPU is available and to `cpu` otherwise (spec §5.7). The CLI tests pass `--device` explicitly.
- **Speed target.** "Much faster" means the M1 config trains on the GPU in at least 5× less time than the M1 CPU run (879 s), so at most 176 s. Its final validation loss stays below 2.0 and within 0.1 of the CPU run's 1.564. CPU and CUDA results are not required to be bit-identical.
- **Benchmark command.** The benchmark is the CLI command `titan bench matmul`. It times with `std::chrono` around `cuda_synchronize()`, a new `core` function, so the CLI needs no CUDA header. Peak FP32 is computed as `2 × SMs × 128 lanes × clock` for sm_86; peak bandwidth is `2 × memory clock × bus width / 8`.
- **Module parity.** Gradient checks stay CPU-only (spec §6). CUDA modules are covered by forward and backward parity tests against the CPU model.

## File structure

| Area | Files |
| --- | --- |
| core | `src/core/device.hpp/.cpp`: `cuda_synchronize()` and more fields in `CudaDeviceInfo` |
| ops/cuda | `src/ops/cuda/{common.cuh,elementwise,reduce,matmul,cublas_handle,softmax,norm,rope,layout,embedding,cross_entropy,adamw}.{hpp,cu}`, `src/ops/cuda/CMakeLists.txt` |
| ops | `src/ops/detail.hpp` (device routing), every `src/ops/<family>.cpp`, `src/ops/matmul.hpp` (`GemmBackend`) |
| nn, model, runtime | `src/model/model.hpp/.cpp` (`to`), `src/runtime/trainer.*` (device) |
| cli | `apps/titan/commands.*`, `apps/titan/main.cpp` |
| tests | `tests/support/parity.hpp/.cpp`, `tests/ops/cuda/<family>_parity_test.cpp`, `tests/nn/module_parity_test.cpp`, `tests/model/model_cuda_test.cpp`, `tests/CMakeLists.txt` |
| docs | `docs/math/cuda-execution.md` (new), the "CUDA implementation" sections, `docs/math/matmul.md` roofline, README "M2 result", AGENTS.md, overview |

---

### Task 1: CUDA backend scaffolding, device routing and elementwise kernels

**Files:**

- Create:
  - `src/ops/cuda/CMakeLists.txt`;
  - `src/ops/cuda/common.cuh` (launch helpers: `grid_for(n, block)`, `kBlock = 256`, a `check_launch()` that wraps `TITAN_CUDA_CHECK(cudaGetLastError())`);
  - `src/ops/cuda/elementwise.hpp/.cu`;
  - `tests/support/parity.hpp/.cpp`;
  - `tests/ops/cuda/elementwise_parity_test.cpp`;
  - `docs/math/cuda-execution.md`.
- Modify:
  - `src/ops/CMakeLists.txt` (`add_subdirectory(cuda)`, link `titan_ops_cuda`);
  - `src/ops/detail.hpp`;
  - `src/ops/elementwise.cpp`;
  - `tests/CMakeLists.txt`;
  - `docs/math/elementwise.md`;
  - `docs/math/README.md`.

**Interfaces:**

- Produces:
  - `enum class detail::Backend { cpu, cuda }`.
  - `detail::Backend detail::route(const char* op, std::initializer_list<const Tensor*>)`. It returns the common device, or throws the mixed-device message.
  - `ops::cuda::` functions with the same signatures as `ops::cpu::` for every elementwise op.
  - `TITAN_REQUIRE_CUDA()` and `TITAN_EXPECT_PARITY(c, g, atol, rtol)`.
  - `titan::test::random_tensor(Shape, std::uint64_t seed)`: a CPU tensor drawn from N(0, 1).

- [ ] **Step 1:** Write `OpsDispatch.MixedDevicesThrow`. It skips without a GPU; otherwise `ops::add(cpu, cuda)` throws a message containing `ops::add: tensors are on different devices (cpu vs cuda)`.
- [ ] **Step 2:** Write parity tests for every elementwise op: `ElementwiseParity.<Op>` for add, add_, add_rowwise_, mul, scale, scale_, silu, silu_backward, gelu and gelu_backward. Each one:
  - builds random CPU inputs;
  - runs the op on the inputs and on `.to(Device::CUDA)` copies;
  - compares the results with `TITAN_EXPECT_PARITY`;
  - covers the three shape classes from Global Constraints.
- [ ] **Step 3:** Run the tests. Expected: FAIL, with `CUDA implementation arrives in M2`.
- [ ] **Step 4:** Implement `route`, the CUDA library and the elementwise kernels. One thread per element in a grid-stride loop. The GELU and SiLU formulas match `ops/cpu` term for term.
- [ ] **Step 5:** Run `scripts/check.ps1`. Expected: `check: OK` and every new test PASS locally.
- [ ] **Step 6:** Write `docs/math/cuda-execution.md` and the CUDA section of `elementwise.md`, then commit with `feat(ops): add cuda backend and elementwise kernels`.

### Task 2: Reductions

**Files:**

- `src/ops/cuda/reduce.hpp/.cu`
- `src/ops/reduce.cpp`
- `tests/ops/cuda/reduce_parity_test.cpp`
- `docs/math/reduce.md`

**Interfaces:**

- Produces `ops::cuda::sum`, `sum_squares`, `row_max`, `sum_leading` and `all_finite`, with the CPU signatures. `sum` and `sum_squares` return `double` to the host.

- [ ] **Step 1:** Write the parity tests `ReduceParity.<Op>`, plus:
  - `RowMaxParity.ManyRows`, on a (70000, 3) tensor;
  - `AllFiniteParity.DetectsNanAndInf`: one NaN or Inf at the last index makes the result false on CUDA.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the kernels:
  - `sum` and `sum_squares` run two passes: per-block double partial sums, then one block sums the partials in a fixed order.
  - `row_max` puts one block per row.
  - `sum_leading` puts one thread per column, accumulating in double over the rows in order.
  - `all_finite` reduces an integer flag.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write the CUDA section of `reduce.md`, then commit with `feat(ops): add cuda reduction kernels`.

### Task 3: Matmul with cuBLAS, naive and tiled kernels

**Files:**

- `src/ops/cuda/matmul.hpp/.cu`
- `src/ops/cuda/cublas_handle.hpp/.cu`
- `src/ops/matmul.hpp/.cpp`
- `tests/ops/cuda/matmul_parity_test.cpp`
- `docs/math/matmul.md`

**Interfaces:**

- Produces:
  - `GemmBackend`, `set_gemm_backend` and `gemm_backend` (Global Constraints).
  - `ops::cuda::matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool ta, bool tb, GemmBackend)`, which computes `out += op(a)·op(b)` for rank 2 to 4 with batch dimensions.
  - `ops::cuda::matmul_naive` and `ops::cuda::matmul_tiled`, which `matmul_acc` selects.

- [ ] **Step 1:** Write `MatmulParity.<Backend>` for each backend. Each covers:
  - all four transpose combinations;
  - a batched rank-4 case (2, 3, 37, 129) × (2, 3, 129, 65);
  - (1, 1) × (1, 1);
  - a 512×512×512 product.

  The tolerance is `atol = 1e-4, rtol = 1e-4`, because float accumulation over K. Also write:
  - `MatmulParity.AccAddsToOut`: `matmul_acc` on a non-zero `out`;
  - `MatmulParity.BackendsAgree`: cuBLAS against tiled on the same inputs;
  - `Matmul.DefaultBackendIsCublas`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the matmul:
  - A process-wide lazily created `cublasHandle_t`, set to `CUBLAS_PEDANTIC_MATH`. It is never destroyed, like the allocator.
  - cuBLAS through `cublasSgemmStridedBatched` with `beta = 1`, using the `Cᵀ = Bᵀ·Aᵀ` mapping.
  - The naive kernel: one thread per output element.
  - The tiled kernel: 16×16 shared-memory tiles, with the bounds guarded on every load.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write the CUDA section of `matmul.md`: the cuBLAS column-major mapping, the tiling, and the arithmetic intensity of each kernel. Task 10 fills in the measured numbers. Commit with `feat(ops): add cuda matmul with cublas, naive and tiled kernels`.

### Task 4: Softmax

**Files:**

- `src/ops/cuda/softmax.hpp/.cu`
- `src/ops/softmax.cpp`
- `tests/ops/cuda/softmax_parity_test.cpp`
- `docs/math/softmax.md`

- [ ] **Step 1:** Write the parity tests:
  - `SoftmaxParity.Softmax`;
  - `SoftmaxParity.Causal`, with `pos_offset` 0 and 5 and Tq ≠ Tk;
  - `SoftmaxParity.Backward`;
  - `SoftmaxParity.ManyRows`, with 70000 rows of 7.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement one block per row: a max pass, an exp-sum pass and a normalize pass, with block tree reductions. The causal mask zeroes the entries with `j > i + pos_offset`, as on the CPU.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write the CUDA section of `softmax.md`, then commit with `feat(ops): add cuda softmax kernels`.

### Task 5: RMSNorm and LayerNorm

**Files:**

- `src/ops/cuda/norm.hpp/.cu`
- `src/ops/norm.cpp`
- `tests/ops/cuda/norm_parity_test.cpp`
- `docs/math/normalization.md`

- [ ] **Step 1:** Write the parity tests for:
  - the forward outputs and the saved statistics;
  - `dx`;
  - the accumulated `dweight` and `dbias` on non-zero accumulators, with 2049 rows.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the kernels:
  - The forward and `dx` put one block per row.
  - `dweight` and `dbias` run as a separate kernel with one thread per column, summing the rows in order in double and then adding to the accumulator.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write the CUDA section of `normalization.md`, then commit with `feat(ops): add cuda normalization kernels`.

### Task 6: RoPE and head-layout ops

**Files:**

- `src/ops/cuda/{rope,layout}.hpp/.cu`
- `src/ops/{rope,layout}.cpp`
- `tests/ops/cuda/{rope,layout}_parity_test.cpp`
- `docs/math/rope.md`
- `docs/math/attention.md`

- [ ] **Step 1:** Write the parity tests:
  - `RopeParity.Forward` and `RopeParity.Backward`, with `pos_offset` 0 and 7, theta 10000, and head dims 2 and 64;
  - `LayoutParity.SwapDims12`, `RepeatKv` and `RepeatKvBackward`, with `n_rep` 1 and 3.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the kernels:
  - RoPE: one thread per (row, pair). The angle is computed in double, as on the CPU, then rounded to float.
  - The layout ops: one thread per output element. `repeat_kv_backward` sums the `n_rep` copies in a fixed order.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write the CUDA sections, then make two commits: `feat(ops): add cuda rope kernels` and `feat(ops): add cuda head-layout kernels`.

### Task 7: Embedding and cross-entropy

**Files:**

- `src/ops/cuda/{embedding,cross_entropy}.hpp/.cu`
- `src/ops/{embedding,cross_entropy}.cpp`
- `tests/ops/cuda/{embedding,cross_entropy}_parity_test.cpp`
- `docs/math/embedding.md`
- `docs/math/cross-entropy.md`

- [ ] **Step 1:** Write the parity tests:
  - `EmbeddingParity.Forward`;
  - `EmbeddingParity.BackwardAccumulates`, with repeated ids and a non-zero accumulator;
  - `CrossEntropyParity.LossAndGrad`, with V = 65 and V = 1000;
  - the two out-of-range tests from Review Focus 4.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the ops:
  - The ids are validated on the host by the dispatch, which already does this for the CPU, then copied to a device buffer from the caching allocator.
  - `embedding_backward` is deterministic: one thread per (vocab row, column) pair, scanning the ids in order and accumulating in double. Its cost is O(V·N·D / threads), which is fine for V ≤ 50k at M2 sizes. The math note states the trade-off against atomics.
  - Cross-entropy puts one block per row and returns the mean loss through a fixed-order reduction.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write the CUDA sections, then make two commits: `feat(ops): add cuda embedding kernels` and `feat(ops): add cuda cross-entropy kernel`.

### Task 8: AdamW update, and remove `require_cpu`

**Files:**

- `src/ops/cuda/adamw.hpp/.cu`
- `src/ops/adamw.cpp`
- `src/ops/detail.hpp`
- `tests/ops/cuda/adamw_parity_test.cpp`
- `docs/math/adamw.md`

- [ ] **Step 1:** Write `AdamWParity.TenSteps`. It runs 10 updates on the CPU and on CUDA from the same state and compares `param`, `m` and `v`.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement the update with one thread per element and the bias correction computed on the host. Delete `detail::require_cpu`, then confirm with a grep that `arrives in M2` is gone from `src/ops`.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write the CUDA section of `adamw.md`, then commit with `feat(ops): add cuda adamw kernel`.

### Task 9: Modules and model on the GPU

**Files:**

- `src/model/model.hpp/.cpp`
- `src/runtime/trainer.hpp/.cpp`
- `src/nn/*` only where a tensor is created on a hard-coded `Device::CPU` that must follow its input
- `tests/nn/module_parity_test.cpp`
- `tests/model/model_cuda_test.cpp`
- `tests/runtime/trainer_test.cpp`
- `docs/math/model.md`

**Interfaces:**

- Produces:
  - `void Model::to(Device)`, which moves every parameter's value and grad.
  - `Device Model::device() const`.
  - `TrainConfig::device` (default `Device::CPU`), which `Trainer` uses to move the model before step 1.

- [ ] **Step 1:** Write the tests:
  - `ModuleParity.<Module>` for Linear, RMSNorm, LayerNorm, SwiGLU, GeluMLP, Attention (GQA) and TransformerBlock. Each runs forward and backward on CPU and CUDA copies of the same module and compares `y`, `dx` and every parameter's gradient.
  - `Model.ToCudaKeepsTiedWeights`.
  - `Model.CudaForwardBackwardMatchesCpu`.
  - `Model.CudaOverfitsOneBatch`, with loss below 0.05 within the M1 test's step budget.
  - `Trainer.CudaSameSeedIsBitReproducible`.
- [ ] **Step 2:** Run them. Expected: FAIL to compile, because `Model::to` does not exist yet.
- [ ] **Step 3:** Implement `Model::to` and the trainer's device handling. Move the tensors that modules create from a hard-coded device so they follow their input's device.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write the device section of `model.md`, then commit with `feat(model): train on cuda`.

### Task 10: CLI device and GEMM flags, matmul benchmark, M2 run

**Files:**

- `src/core/device.hpp/.cpp`
- `apps/titan/commands.*`
- `apps/titan/main.cpp`
- `tests/CMakeLists.txt`
- `docs/math/matmul.md`
- `README.md`
- `AGENTS.md`
- `docs/architecture/overview.md`

**Interfaces:**

- Produces:
  - `void cuda_synchronize()` in `core/device.hpp`. It throws without a GPU.
  - `CudaDeviceInfo` gains the fields `int sm_count`, `int clock_khz`, `int memory_clock_khz` and `int memory_bus_width_bits`.
  - `titan train ... [--device cpu|cuda] [--gemm cublas|naive|tiled]`.
  - `titan bench matmul [--sizes 256,512,1024,2048,4096] [--repeat 10]`. It prints the device's peak FP32 GFLOP/s and bandwidth, then one line per (size, backend) with ms, GFLOP/s and % of peak.

- [ ] **Step 1:** Write the CTest entries:
  - Add `--device cpu` to the existing CPU CLI tests, and remove `cli_train_cuda_rejected`.
  - `cli_train_cuda_smoke`: the expected output matches `final val_loss`, with `SKIP_REGULAR_EXPRESSION "no CUDA device available"`.
  - `cli_train_gemm_needs_cuda`.
  - `cli_bench_matmul_smoke`: `--sizes 64 --repeat 1`. It matches `tiled`, and skips without a GPU.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the flags, the default device and the bench command. Validate every flag before anything is written to `--out`.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Run the M2 training and the benchmark:
  - Run `titan train --config configs/tiny-shakespeare.json --data data/tinyshakespeare/input.txt --out runs/m2-shakespeare --device cuda`, then `titan bench matmul`.
  - Record the run time, the validation-loss trajectory and the sample in the README's "M2 result" section, next to M1.
  - Record the benchmark table and the roofline analysis in `matmul.md`.
  - Check the speed target in Decisions.
- [ ] **Step 6:** Update AGENTS.md (commands, with `src/ops/cuda/` moved to the current layout) and the overview, then commit with `feat(cli): train on cuda and benchmark matmul`.
