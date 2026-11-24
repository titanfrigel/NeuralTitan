---
status: accepted
date: 2026-11-24
packages: [tokenizers]
---

# Pinning the reference tokenizer package for Unicode probing

## Context and problem

Milestone M4 must split text exactly as SmolLM2's reference tokenizer does. That tokenizer uses Unicode 16 classes, but Python's `unicodedata` is Unicode 15.1, so tables built from it differ from the reference. `tools/gen_unicode_tables.py` therefore builds `src/tokenizer/unicode_tables.*` by probing Hugging Face `tokenizers`, the Rust tokenizer library that `transformers` already installs (ADR 0004), on every Unicode scalar value. The generated tables depend on that library's exact version, so the version must be pinned directly instead of left to `transformers`' dependency range.

## Decision drivers

- Exactness: the tables must reproduce the reference pre-tokenizer bit for bit.
- Reproducibility: regenerating the tables must give the same files.
- No new runtime dependency: the C++ build stays free of third-party libraries apart from cuBLAS (ADR 0007).
- Development-only tooling in the isolated `tools/.venv`, as in ADR 0004.

## Considered options

| Criterion | A. Pin `tokenizers` and probe it | B. Python `unicodedata` | C. Download `UnicodeData.txt` 16.0 |
| --- | --- | --- | --- |
| Fit | Matches the reference by construction, including the regex engine's class quirks | Unicode 15.1: thousands of letters and numbers differ | Right Unicode version, but not the reference's regex engine (Oniguruma and Rust disagree on some digits) |
| Packages | `tokenizers` 0.23.3 (already installed with `transformers` 5.23.0) | none | a pinned data file |
| License | Apache-2.0 | PSF | Unicode License v3 |
| Cost and exit | Free; regenerate needs the venv; remove the pin to exit | Free | Free; a download to pin and verify |

Chosen: A. B is rejected because it cannot match the reference. C is rejected because it still differs from the reference's classes.

## Decision outcome

Chosen: pin `tokenizers==0.23.3` in `tools/requirements-dev.txt`, next to `transformers==5.23.0`. `tools/gen_unicode_tables.py` asserts this version and records it in the generated tables as `kUnicodeSource`.

## Consequences

- The generated tables are committed, so building and testing never need Python.
- Upgrading `tokenizers` means regenerating the tables and the SmolLM2 fixtures together.
- The tables inherit any quirk of the reference, which is the target behaviour for M4.

## Links

- [ADR 0004: golden reference tooling](0004-golden-reference-tooling.md)
- [tokenizers on PyPI](https://pypi.org/project/tokenizers/)
- [docs/math/bpe.md](../math/bpe.md), section on the Unicode classes
