"""Generates the Hugging Face Llama fixtures in tests/fixtures/.

The reference is Hugging Face transformers. With --tiny the script writes two
tiny random Llama models, `tied/` and `untied/`, in tests/fixtures/tiny_llama/,
each a directory as `save_pretrained` writes it plus a tokenizer:

- config.json and model.safetensors, as written by
  `LlamaForCausalLM.save_pretrained(safe_serialization=True)` from a model
  initialized with torch.manual_seed(0), its RMSNorm weights set to
  1 + 0.1 * N(0, 1) (randomize_norms), and converted to BF16 (TINY_CONFIG,
  with tie_word_embeddings True or False);
- tokenizer.json, a copy of tests/fixtures/bpe/trained.json (vocab 512,
  <|endoftext|> at id 0).

With --golden the script runs Hugging Face transformers (FP32, CPU, eager
attention) on the tiny models and on SmolLM2-135M (pinned revision, downloaded
to models/SmolLM2-135M/ if missing) and writes the reference outputs that the
C++ parity tests compare against:

- tests/fixtures/tiny_llama/golden.{json,safetensors}: for each of `tied` and
  `untied` and each of three prompts, the full logits at every prompt position
  and the top-2 logit gap there, plus the greedy continuation;
- tests/fixtures/smollm2/golden.{json,safetensors}: for each of four prompts,
  the last-position logits, the logits of 1024 sampled vocabulary ids at every
  position and the top-2 gap there, plus the greedy continuation.

The greedy gaps are the differences of the two largest FP32 logits at each
generated step (float64 in the JSON), taken from the raw output logits. A
generated end-of-text token is not listed in `greedy`, but its step's gap is
the last entry of `greedy_gaps`.

With --all the script runs --tiny and then --golden. The output is
deterministic: a re-run writes byte-identical files.

Run from the repository root, in the tools venv:

    tools/.venv/Scripts/python -I tools/make_llama_fixtures.py --all
"""

import argparse
import json
import pathlib
import shutil
import sys

import tokenizers
import torch
import transformers
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
TINY_DIR = ROOT / "tests" / "fixtures" / "tiny_llama"
TINY_TOKENIZER = ROOT / "tests" / "fixtures" / "bpe" / "trained.json"
SMOLLM2_DIR = ROOT / "tests" / "fixtures" / "smollm2"
MODEL_DIR = ROOT / "models" / "SmolLM2-135M"

# Same pin as tools/make_tokenizer_fixtures.py.
REPO_ID = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors",
               "tokenizer.json"]

TOLERANCE = 1e-3
EOS_ID = 0
SMOLLM2_VOCAB = 49152
SMOLLM2_SAMPLES = 1024
SMOLLM2_NEW_TOKENS = 32
TINY_NEW_TOKENS = 16
SMOLLM2_PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):\n",
    "Café au lait, s'il vous plaît. ",
]
TINY_PROMPTS = ["to be or not", "the king", "ROMEO:\n"]

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


def randomize_norms(model) -> None:
    """Sets every RMSNorm weight to 1 + 0.1 * N(0, 1).

    A fresh model's norm weights are all 1, which would make
    input_layernorm, post_attention_layernorm and model.norm interchangeable
    and hide a mix-up of them in the loader. The draws come from their own
    generator, seeded 1, in named_parameters order, so the other weights are
    those of torch.manual_seed(0) as before.
    """
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            is_norm = name.endswith("layernorm.weight")
            if is_norm or name == "model.norm.weight":
                noise = torch.randn(param.shape, generator=gen)
                param.copy_(1.0 + 0.1 * noise)


def write_tiny(name: str, tied: bool) -> None:
    out = TINY_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    config = transformers.LlamaConfig(**TINY_CONFIG, tie_word_embeddings=tied)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    randomize_norms(model)
    model = model.to(torch.bfloat16)
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


def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def top2_gap(logits: torch.Tensor) -> torch.Tensor:
    """Largest minus second-largest logit along the last dimension."""
    top = torch.topk(logits, 2, dim=-1).values
    return top[..., 0] - top[..., 1]


def reference_run(model, ids: list[int], new_tokens: int):
    """Logits at every prompt position and the greedy continuation.

    Returns (logits [T, vocab], greedy ids without a final end-of-text,
    greedy gaps as float64). The gaps come from the raw output logits, not the
    processed scores.
    """
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=new_tokens,
            eos_token_id=EOS_ID,
            pad_token_id=EOS_ID,
            output_logits=True,
            return_dict_in_generate=True,
        )
    generated = out.sequences[0, len(ids):].tolist()
    gaps = [float(top2_gap(step[0]).to(torch.float64)) for step in out.logits]
    assert len(gaps) == len(generated)
    if generated and generated[-1] == EOS_ID:
        generated = generated[:-1]
    return logits, generated, gaps


def load_reference(path: pathlib.Path):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager")
    return model.eval()


def make_tiny_golden() -> None:
    torch.manual_seed(0)
    golden = {"tolerance": TOLERANCE}
    tensors = {}
    for name in ("tied", "untied"):
        model = load_reference(TINY_DIR / name)
        tok = tokenizers.Tokenizer.from_file(
            str(TINY_DIR / name / "tokenizer.json"))
        entries = []
        for i, text in enumerate(TINY_PROMPTS):
            ids = tok.encode(text, add_special_tokens=False).ids
            logits, greedy, gaps = reference_run(model, ids, TINY_NEW_TOKENS)
            tensors[f"{name}.p{i}.logits"] = logits.contiguous()
            tensors[f"{name}.p{i}.top2_gap"] = top2_gap(logits).contiguous()
            entries.append({"text": text, "ids": ids, "greedy": greedy,
                            "greedy_gaps": gaps})
            print(f"tiny/{name} p{i} {text!r}\n  ids={ids}\n  greedy={greedy}"
                  f"\n  text={tok.decode(ids + greedy, skip_special_tokens=False)!r}")
        golden[name] = entries
    write_json(TINY_DIR / "golden.json", golden)
    save_file(tensors, str(TINY_DIR / "golden.safetensors"))
    print("wrote tests/fixtures/tiny_llama/golden.{json,safetensors}")


def ensure_smollm2() -> None:
    if all((MODEL_DIR / f).exists() for f in MODEL_FILES):
        return
    from huggingface_hub import hf_hub_download
    for f in MODEL_FILES:
        hf_hub_download(REPO_ID, f, revision=REVISION, local_dir=str(MODEL_DIR))


def make_smollm2_golden() -> None:
    ensure_smollm2()
    SMOLLM2_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    sample_ids = torch.randperm(
        SMOLLM2_VOCAB, generator=torch.Generator().manual_seed(0)
    )[:SMOLLM2_SAMPLES].sort().values
    tok = tokenizers.Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    auto = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_reference(MODEL_DIR)
    entries = []
    tensors = {}
    for i, text in enumerate(SMOLLM2_PROMPTS):
        ids = tok.encode(text, add_special_tokens=False).ids
        assert ids == auto(text, add_special_tokens=False)["input_ids"], text
        logits, greedy, gaps = reference_run(model, ids, SMOLLM2_NEW_TOKENS)
        tensors[f"p{i}.last"] = logits[-1].contiguous()
        tensors[f"p{i}.sampled"] = logits[:, sample_ids].contiguous()
        tensors[f"p{i}.top2_gap"] = top2_gap(logits).contiguous()
        entries.append({"text": text, "ids": ids, "greedy": greedy,
                        "greedy_gaps": gaps})
        print(f"smollm2 p{i} {text!r}\n  ids={ids}\n  greedy={greedy}\n"
              f"  text={tok.decode(greedy, skip_special_tokens=False)!r}")
    write_json(SMOLLM2_DIR / "golden.json", {
        "revision": REVISION,
        "tolerance": TOLERANCE,
        "sample_ids": sample_ids.tolist(),
        "prompts": entries,
    })
    save_file(tensors, str(SMOLLM2_DIR / "golden.safetensors"))
    print("wrote tests/fixtures/smollm2/golden.{json,safetensors}")


def make_golden() -> None:
    torch.use_deterministic_algorithms(True)
    make_tiny_golden()
    make_smollm2_golden()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="write the tiny tied and untied models to tests/fixtures/tiny_llama/",
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="write golden.json and golden.safetensors for the tiny models "
        "(which --tiny must have written) and for SmolLM2-135M",
    )
    parser.add_argument("--all", action="store_true",
                        help="run --tiny, then --golden")
    args = parser.parse_args()
    if not (args.tiny or args.golden or args.all):
        parser.error("nothing to do: pass --tiny, --golden or --all")
    if args.tiny or args.all:
        make_tiny()
    if args.golden or args.all:
        make_golden()


if __name__ == "__main__":
    main()
