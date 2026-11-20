---
status: accepted
date: 2026-11-19
packages: [torch, transformers, safetensors]
---

# Golden reference tooling for pretrained-model tests

## Context and problem

Milestones M4 and M5 require proof that our BPE tokenizer and Llama implementation reproduce SmolLM2-135M exactly: identical token ids, logits within 1e-3, and matching greedy generations. That proof needs reference outputs from an independent, authoritative implementation. The tool runs only on the author's machine, to produce fixture files. It is never part of the C++ build (ADR 0003).

## Decision drivers

- Independence: the reference must not share our own understanding or mistakes.
- Authority: SmolLM2 is defined against the Hugging Face implementation.
- FP32 support, matching our loader's BF16-to-FP32 upcast.
- Isolation from the library build and from the system Python.

## Considered options

| Criterion | A. Python + PyTorch + transformers | B. llama.cpp | C. Own NumPy reimplementation |
| --- | --- | --- | --- |
| Fit | The reference implementation SmolLM2 is defined against; runs in FP32 | Conversion needs Python packages anyway; numerics may differ | Shares our own understanding, so cannot catch our mistakes |
| Packages | `torch` 2.15.0, `transformers` 5.23.0, `safetensors` 0.8.0 | C++ build plus Python conversion | `numpy`, `safetensors` |
| Health | `transformers` ~28M downloads/week; latest releases 2026-10-28 (torch) and 2026-11-13 (transformers); no OSV advisories at these versions | Active (unverified) | Mature (unverified) |
| License | Apache-2.0 (torch includes BSD/MIT parts) | MIT (unverified) | BSD / Apache-2.0 (unverified) |
| Cost and exit | Free; isolated venv, removed by deleting it | Free; a second toolchain | Free; most effort |

Chosen: A. B is rejected for weaker numerical equivalence and an extra toolchain. C is rejected because it is not independent.

## Decision outcome

Chosen: Python with `torch` (CPU build), `transformers` and `safetensors`. They are installed in an isolated virtual environment under `tools/`, with versions pinned by the package manager. Scripts there dump tokenizer ids, logits and greedy generations to fixture files committed under the test data directory.

## Consequences

- Large download (CPU-only PyTorch is on the order of hundreds of MB), needed only when regenerating fixtures.
- Fixtures are committed, so building and testing NeuralTitan never requires Python.
- Requires a local Python 3.11+ (3.13.7 installed).
- Version bumps of these packages are optional; fixtures only need regenerating if the reference changes.

## Links

- [transformers on PyPI](https://pypi.org/project/transformers/) (fetched 2026-11-19)
- [torch on PyPI](https://pypi.org/project/torch/) (fetched 2026-11-19)
- [safetensors on PyPI](https://pypi.org/project/safetensors/) (fetched 2026-11-19)
- [OSV](https://osv.dev/) query for each package at the listed versions (fetched 2026-11-19)
- [ADR 0003: Third-party libraries in the build](0003-no-third-party-libraries.md)
