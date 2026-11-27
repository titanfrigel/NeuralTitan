"""Generates the Hugging Face Llama fixtures in tests/fixtures/tiny_llama/.

The reference is Hugging Face transformers. With --tiny the script writes two
tiny random Llama models, `tied/` and `untied/`, each a directory as
`save_pretrained` writes it plus a tokenizer:

- config.json and model.safetensors, as written by
  `LlamaForCausalLM.save_pretrained(safe_serialization=True)` from a model
  initialized with torch.manual_seed(0) and converted to BF16 (TINY_CONFIG,
  with tie_word_embeddings True or False);
- tokenizer.json, a copy of tests/fixtures/bpe/trained.json (vocab 512,
  <|endoftext|> at id 0).

The output is deterministic: a re-run writes byte-identical files.

Run from the repository root, in the tools venv:

    tools/.venv/Scripts/python -I tools/make_llama_fixtures.py --tiny
"""

import argparse
import pathlib
import shutil

import torch
import transformers

ROOT = pathlib.Path(__file__).resolve().parent.parent
TINY_DIR = ROOT / "tests" / "fixtures" / "tiny_llama"
TINY_TOKENIZER = ROOT / "tests" / "fixtures" / "bpe" / "trained.json"

# The tiny model: two layers, grouped-query attention (4 heads, 2 KV heads)
# and a vocabulary that matches the trained BPE fixture. initializer_range is
# large so that the weights, and later the logits, are far from zero.
TINY_CONFIG = dict(
    vocab_size=512,
    hidden_size=64,
    intermediate_size=160,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    rope_theta=100000.0,
    rms_norm_eps=1e-5,
    initializer_range=0.2,
    bos_token_id=0,
    eos_token_id=0,
)


def write_tiny(name: str, tied: bool) -> None:
    out = TINY_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    config = transformers.LlamaConfig(**TINY_CONFIG, tie_word_embeddings=tied)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(out, safe_serialization=True)
    # save_pretrained also writes generation_config.json, which only restates
    # bos_token_id and eos_token_id; the loader reads config.json alone.
    (out / "generation_config.json").unlink()
    # On Windows save_pretrained writes config.json with CRLF line endings;
    # the repository stores LF (.gitattributes), so a re-run matches a fresh
    # checkout byte for byte. Nothing else in the file changes.
    config_path = out / "config.json"
    config_path.write_bytes(config_path.read_bytes().replace(b"\r\n", b"\n"))
    shutil.copyfile(TINY_TOKENIZER, out / "tokenizer.json")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


def make_tiny() -> None:
    write_tiny("tied", tied=True)
    write_tiny("untied", tied=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="write the tiny tied and untied models to tests/fixtures/tiny_llama/",
    )
    args = parser.parse_args()
    if not args.tiny:
        parser.error("nothing to do: pass --tiny")
    if args.tiny:
        make_tiny()


if __name__ == "__main__":
    main()
