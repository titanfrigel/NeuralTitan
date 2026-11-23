# CLI fixtures

Hand-written inputs for the `cli_*` CTest entries. They are not generated golden fixtures, so they are edited by hand rather than regenerated.

- `tiny_corpus.txt`: a few public-domain lines of Shakespeare.
- `tiny-config.json`: a model and training config small enough to train in seconds.
- `bad-config.json`: the same with 4 heads and `n_kv_heads` 3, which must be rejected.
- `checkpoint-config.json`: `tiny-config.json` with `checkpoint_interval` 3, for the save and resume tests.
- `no_newline_corpus.txt`: one line without a newline character; training must still succeed (the sample prompt falls back to token 0).
- `short_corpus.txt`: too few tokens for the training and validation splits; training must fail before writing anything.
