---
status: accepted
date: 2026-11-19
packages: [clang-format, gitleaks, markdownlint-cli2, lychee, actions/checkout, actions/cache, actions/upload-artifact, github/codeql-action, ossf/scorecard-action]
---

# Repository tooling baseline: format, lint, secrets, docs checks and CI

## Context and problem

Milestone M0 sets up the repository tooling the setup record calls for (Docs full, Tests standard, Public yes, Maintained no, Versioned no). The project is C++20/CUDA 12.6 on Windows, with no package manager. It needs:

- one formatter;
- a lint strictness;
- a secret scan before commit and in CI;
- docs checks;
- a CI workflow that runs the single `check` entry point (`scripts/check.ps1`).

The tooling has to give the same result on the developer machine (VS 2026 with MSVC toolset 14.38) and on GitHub's hosted runners. ADR 0003 and ADR 0007 keep the build's own dependencies minimal, and this ADR keeps CI the same way.

Facts verified on 2026-11-19:

- **Runner.** The `windows-2025` runner image ships Visual Studio 2022 17.14, which has MSVC toolset 14.44, and has no CUDA.
- **CUDA and MSVC.** CUDA 12.6 accepts `_MSC_VER` 1910 to 1949 (`crt/host_config.h`), so 14.44 is supported and only 14.50 and later is rejected.
- **CUDA installer.** NVIDIA's network installer for CUDA 12.6.3 can install only selected subpackages. 12.6.3 is the release of the local `nvcc` V12.6.85.
- **clang-format.** VS 2026 bundles clang-format 20.1.8. The runner's VS 2022 bundles an older LLVM. PyPI publishes `clang-format` 20.1.8.
- **gitleaks.** The current release is v8.30.1. The documented hook command is `gitleaks git --pre-commit --redact --staged --verbose`.
- **Actions.** The current releases are `actions/checkout` v7.0.1, `actions/cache` v6.1.0, `actions/upload-artifact` v7.0.2 and `github/codeql-action` v4.39.3. `ossf/scorecard-action` v2.6.0 was released on the day of this decision, so v2.5.0 is pinned.

## Decision drivers

- The same formatting and the same checks locally and in CI.
- Few third-party actions to trust and keep pinned in a public repository.
- No new runtime dependency. Every tool here is development-only.
- Fast feedback: `check` stays a single command.

## Considered options

- **Formatter.**
  - Chosen: clang-format 20.1.8 pinned from PyPI.
  - Rejected: the VS-bundled clang-format, because its major version differs between local and CI.
  - Rejected: no formatter.
- **Lint.**
  - Chosen: compiler warnings as errors.
  - Rejected: adding clang-tidy, which is slower and noisy on CUDA headers under Windows. It can be revisited later.
- **Hook mechanism.**
  - Chosen: a plain Git hook in `.githooks/`.
  - Rejected: the `pre-commit` framework, which is one more Python tool to install and pin.
- **CI actions.**
  - Chosen: GitHub's own actions plus `ossf/scorecard-action`, with tools downloaded from their vendors.
  - Rejected: marketplace actions (`Jimver/cuda-toolkit`, `amannn/action-semantic-pull-request`, `gitleaks/gitleaks-action`, `DavidAnson/markdownlint-cli2-action`, `lycheeverse/lychee-action`). They mean five more third-party actions to trust and pin.

## Decision outcome

- **Format.** `clang-format==20.1.8` is installed from a pinned `tools/requirements-dev.txt` into `tools/.venv`. `.clang-format` holds the style. `check` fails on unformatted `*.hpp`, `*.cpp`, `*.cu` and `*.cuh` files under `src/`, `apps/` and `tests/`.
- **Lint.** Compiler warnings are errors: MSVC `/W4 /WX`, and nvcc `-Werror all-warnings` with `/W4 /WX` forwarded to the host compiler.
- **Secrets.**
  - Locally, `.githooks/pre-commit` runs `gitleaks git --pre-commit --redact --staged --verbose` with the installed gitleaks (v8.30.1). It is enabled once with `git config core.hooksPath .githooks`.
  - In CI, the gitleaks v8.30.1 release binary is downloaded and verified against its published checksums, then scans the PR's commits.
- **Docs checks.**
  - `markdownlint-cli2` 0.24.1 runs through `npx` in CI.
  - The `lychee` release binary, verified by checksum, checks links on PRs and on a weekly schedule.
  - The generated `docs/adr/README.md` and `.github/scripts/senior-standards/` are excluded.
- **CI workflows.** All run on the `windows-2025` or `ubuntu-latest` runners, with `permissions: {}` at workflow level and actions pinned by full commit SHA.
  - `ci`: the CUDA 12.6.3 subpackages `nvcc`, `cudart`, `cublas` and `cublas_dev` are installed from NVIDIA's network installer and cached with `actions/cache`. MSVC is the runner's 14.44 toolset, selected through an override in `scripts/dev-env.ps1`. The job runs `scripts/check.ps1`, and CUDA tests skip themselves.
  - `pr-title`: an inline Conventional Commits regex check on the PR title, with no third-party action.
  - `gitleaks`.
  - `dependency-check`: the vendored senior-standards check.
  - `docs`.
  - `scorecard`: `ossf/scorecard-action` v2.5.0, with results uploaded through `github/codeql-action/upload-sarif`.
- **CodeQL.** GitHub default setup for C/C++, as a repository setting.
- **Repository settings.** Asked one by one after the first push:
  - the `main` ruleset (required status checks added after CI reports once);
  - secret scanning with push protection;
  - CodeQL default setup;
  - read-only default workflow token.

## Consequences

- **Formatting needs the venv.** Developers create `tools/.venv` once and install `tools/requirements-dev.txt`. `check` reports the missing formatter with the install command.
- **Compiler mismatch.** CI builds with MSVC 14.44, the developer with 14.38. Both are supported by CUDA 12.6, but a toolset-specific issue could appear in only one of them. ADR 0002's local pin is unchanged.
- **CI and the GPU.** CI has no GPU. CUDA parity tests run only locally, and pull requests say so (CONTRIBUTING.md).
- **CUDA setup time.** The CUDA install adds a few minutes to a cold CI run; the cache absorbs it on later runs.
- **Pin updates.** Maintained is `no`, so there is no Renovate. Action SHAs and tool versions are bumped by hand, through a new ADR when a tool changes.

## Links

- [ADR 0002: Language and build toolchain](0002-language-and-build-toolchain.md)
- [ADR 0007: cuBLAS as the default GEMM](0007-cublas-default-gemm.md)
- [GitHub runner image: Windows Server 2025](https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-Readme.md)
- [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [clang-format on PyPI](https://pypi.org/project/clang-format/)
- [gitleaks](https://github.com/gitleaks/gitleaks)
- [markdownlint-cli2](https://github.com/DavidAnson/markdownlint-cli2)
- [lychee](https://github.com/lycheeverse/lychee)
- [OpenSSF Scorecard action](https://github.com/ossf/scorecard-action)
