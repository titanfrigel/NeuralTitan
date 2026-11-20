---
status: accepted
date: 2026-11-19
packages: []
---

# Record architecture decisions

## Context and problem

NeuralTitan is a learning project whose value lies in understanding why it is built the way it is. Decisions about the toolchain, dependencies and structure need a durable, reviewable record, and dependencies are approved only through such records.

## Decision drivers

- Decisions and their rationale must stay readable long after they are made.
- Dependencies may only enter the project through an explicit, recorded approval.
- Low ceremony for a solo project.

## Considered options

| Option | Verdict |
| --- | --- |
| Markdown ADRs in `docs/adr/` with `status`/`date`/`packages` frontmatter | Chosen: versioned with the code, reviewable in PRs, machine-readable package approvals. |
| Decisions in the design spec only | Rejected: specs are rewritten; history and per-decision status get lost. |
| No formal records | Rejected: no audit trail for dependency approvals. |

## Decision outcome

Chosen: Markdown ADRs in `docs/adr/NNNN-kebab-title.md`, append-only, with frontmatter keys `status`, `date` and `packages`. Only `accepted` ADRs approve packages. `docs/adr/README.md` is a generated index.

## Consequences

- Every dependency, toolchain or structural decision gets an ADR before it takes effect.
- Accepted ADRs are never rewritten; a changed decision supersedes the old one.

## Links

- [ADR index](README.md)
