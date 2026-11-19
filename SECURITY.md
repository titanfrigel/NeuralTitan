# Security policy

NeuralTitan is a local, offline learning project. Its main security surface is the parsing of downloaded files: model weights (safetensors), `tokenizer.json`, `config.json`, checkpoints and token files.

## Reporting a vulnerability

Report suspected vulnerabilities privately. Do not open a public issue.

- Contact: [GitHub private vulnerability reporting](https://github.com/titanfrigel/NeuralTitan/security/advisories/new).
- Include a description, reproduction steps (a sample file if a parser is involved), the commit you tested and the impact you see.
- Expect an acknowledgement within 14 days, on a best-effort basis since this is a personal project, and status updates as the fix progresses.

## Supported versions

There are no releases; only the latest `main` receives fixes.

| Version | Supported |
| --- | --- |
| `main` (latest) | Yes |
| Older commits | No |

## Disclosure

Please allow up to 90 days for a fix before public disclosure. Fixed issues are disclosed through a GitHub security advisory that credits the reporter, unless they prefer otherwise.
