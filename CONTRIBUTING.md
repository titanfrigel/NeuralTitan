# Contributing

## Workflow

- Work on a feature branch named `<type>/<short-topic>`, where `<type>` is a Conventional Commit type (`feat`, `fix`, `docs`, `chore`, ...), for example `feat/rmsnorm`.
- Open pull requests into `main`.
- One purpose per pull request, with a description of what changed and why. The pull request title follows Conventional Commits; CI checks it.
- Docs are updated in the same pull request as the code: README, requirements, architecture overview, decision records and the math notes in `docs/math/` where they are affected.
- CI has no GPU. Run the CUDA tests locally before merging and say so in the pull request.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, for example `feat(ops): add tiled matmul kernel` or `fix(io): reject truncated safetensors header`. Keep commits small and single-purpose.

## Language

All code, comments, commit messages, documentation and pull requests are written in English.

## Checks

Run the `check` entry point before pushing; CI runs the same command. It builds everything, runs the tests, and checks layer rules, docs and secrets. It is created in milestone M0 (see `docs/plans/`), and this section will give its exact command once it exists.

## Tests

Tests are written with each feature, in the same pull request. For this project that means:

- every new op gets a CPU-vs-CUDA parity test, including shapes that are not multiples of tile sizes;
- every new module gets a finite-difference gradient check of its backward pass;
- every parser of external files gets tests with malformed and truncated input.

Coverage is not enforced.

## Hooks

A secret scanner (gitleaks) runs as a pre-commit hook and again in CI. The hook setup is added with the repository tooling in milestone M0, and this section will give the one-time install command then.

## Dependencies

The library links only the C++ standard library, the CUDA runtime and cuBLAS (for matrix multiplication only; see [ADR 0007](docs/adr/0007-cublas-default-gemm.md)). Anything else is out of scope by design.

Any new direct dependency, including development tooling under `tools/`, needs an accepted ADR (architecture decision record) in `docs/adr/` that lists it, in the same pull request. CI runs a dependency check and fails the pull request without one.

Accepted ADRs are append-only: supersede them with a new ADR, never rewrite them. Use `docs/adr/README.md` for the index and the existing records as the format.

## Review checklist

- `check` passes, and new behavior has tests as described above.
- CUDA tests pass locally.
- Layer rules in `docs/architecture/overview.md` still hold.
- New modules and op families have a math note in `docs/math/`.
