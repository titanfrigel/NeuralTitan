# CLI fixtures

Hand-written inputs for the `cli_*` CTest entries. They are not generated golden fixtures, so they are edited by hand rather than regenerated.

- `tiny_corpus.txt`: a few public-domain lines of Shakespeare.
- `tiny-config.json`: a model and training config small enough to train in seconds.
- `bad-config.json`: the same with 4 heads and `n_kv_heads` 3, which must be rejected.
- `checkpoint-config.json`: `tiny-config.json` with `checkpoint_interval` 3, for the save and resume tests.
- `no_newline_corpus.txt`: one line without a newline character; training must still succeed (the sample prompt falls back to token 0).
- `short_corpus.txt`: too few tokens for the training and validation splits; training must fail before writing anything.
- `bad-tokenizer.json`: a BPE tokenizer in Hugging Face's `tokenizer.json` layout whose merge `a b` yields `ab`, which is not in the vocabulary; loading it must fail with an error naming `model.merges[0]`.
- `bad-hf-model/config.json`: a copy of `tests/fixtures/tiny_llama/tied/config.json` with `"rope_scaling": {"type": "linear"}` added; `titan generate --hf-model` must reject it with an error naming `rope_scaling`. The directory holds no `tokenizer.json` or `model.safetensors`: `load_hf_model` validates `config.json` before it opens any other file, and the test relies on that order.

The resume-failure tests edit token files at test time with `tests/tools/edit_token_file.cpp`; no edited token file is stored here.
