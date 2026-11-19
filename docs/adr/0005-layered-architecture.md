---
status: accepted
date: 2026-11-19
packages: []
---

# Architecture style

## Context and problem

NeuralTitan is a library of composable LLM building blocks, read and reviewed by its author to learn how each part works and how the parts combine. Every operation needs a CPU and a CUDA implementation behind one interface. The structure must keep the math easy to follow from a model's forward pass down to a kernel, and dependency directions must stay enforceable in CI.

## Decision drivers

- Readability: following a forward pass should mean following direct calls, not adapters.
- Swappable compute devices (CPU and CUDA) for every operation.
- Low ceremony: solo project, no server, no bindings, fixed inputs and outputs (non-goals in the requirements).
- Transfer: the structure should resemble real LLM codebases.
- Enforceable dependency rules without third-party tools (ADR 0003).

## Considered options

| Option | Verdict |
|---|---|
| Layered, with swappable device backends inside `ops` | Chosen. Calls follow the math directly. Only the device boundary gets an interface, the one place it pays off. The structure resembles PyTorch, llm.c and llama.cpp. |
| Hexagonal (ports and adapters) | Rejected. It makes every input and output swappable, but it adds 5–8 interfaces and mapping code that sit between the reader and the math. The scenarios it serves (servers, bindings, several data sources) are non-goals. |
| Clean / onion | Rejected: built for long-lived business-domain systems; the most ceremony. |
| Feature-sliced | Rejected: a library of composable layers is not a set of independent features. |

## Decision outcome

Chosen: a layered architecture in which each layer depends only on the layers below it:

```
cli → runtime → model → nn / optim → ops → core
```

Two helper layers sit alongside: `tokenizer` and `io`. Device polymorphism lives only in `ops`: one device-dispatching interface, with independent `cpu` and `cuda` implementations behind it.

The layer rules are listed in the architecture overview and enforced two ways:
1. each layer is its own CMake library target, linked only to the layers it may use;
2. an include-direction check script runs in the project's `check` step.

## Consequences

- A forward pass reads top to bottom as plain calls.
- Input and output code (file formats, data loading) is not pluggable. Supporting another weight format means editing the loader. If that becomes frequent, a single interface can be introduced there without restructuring.
- Testing upper layers uses small real inputs (tiny models and corpora) rather than fakes.
- Layer leaks are prevented by CI, not by convention alone.

## Links

- [Architecture overview](../architecture/overview.md)
- [ADR 0003: Third-party libraries in the build](0003-no-third-party-libraries.md)
