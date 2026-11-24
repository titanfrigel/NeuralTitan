"""Generates the SmolLM2 tokenizer golden fixtures in tests/fixtures/smollm2/.

The reference is Hugging Face tokenizers with SmolLM2-135M's tokenizer.json at
a pinned revision. The script:

1. downloads models/SmolLM2-135M/tokenizer.json if it is missing;
2. copies it, unmodified, to tests/fixtures/smollm2/tokenizer.json;
3. writes encode.json: the reference ids and decoded text of FIXTURE_STRINGS;
4. writes corpus.json, if data/tinyshakespeare/input.txt exists: its byte
   count, token count and the CRC-32 (zlib.crc32) of its ids as little-endian
   int32.

Run from the repository root, in the tools venv:

    tools/.venv/Scripts/python -I tools/make_tokenizer_fixtures.py
"""

import json
import pathlib
import shutil
import struct
import zlib

import tokenizers

REPO_ID = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "SmolLM2-135M"
OUT_DIR = ROOT / "tests" / "fixtures" / "smollm2"
CORPUS = "data/tinyshakespeare/input.txt"

# Each string pins one behaviour of the pre-tokenizer, the merges or the
# added-token split. Order matters: the tests refer to cases by index.
FIXTURE_STRINGS = [
    "",
    "Hello world",
    "Hello world 12345 <|endoftext|> héllo",
    "I'm don't we're they've she'll he'd it's",
    "I'M DON'T",
    "x  ' 12ab",
    "a\n\nb  \n  c   ",
    "\t\ttabs\r\n",
    "日本語のテキスト",
    "\U0001f642\U0001f44d\U0001f3fd emoji",
    "١٢٣ ٤٥ Ⅻ ½",
    "naïve café résumé "
    "Ελληνικά "
    "Русский",
    "مرحبا "
    "بالعالم",
    "a<|im_start|>b<|im_end|>c",
    "<|endoftext|><|endoftext|>",
    "<|endoftext",
    "def f(x):\n    return x**2  # comment\n",
    "    leading spaces",
    # No-break space, ideographic space (U+3000), line separator (U+2028).
    " nbsp　ideographic sep",
    "Mixed123abc456",
    "$$$ !!! ??? ... ---",
    "https://example.com/path?q=1&r=2",
    "a" * 200,
    "First Citizen:\nBefore we proceed any further, hear me speak.",
    # The symbol of byte 0x04 is not in the vocabulary: the byte is dropped
    # before merging, so the two "!" merge into one token.
    "!\x04!",
    # The contractions are case-sensitive: "'M" and "'RE" are not split off,
    # so "Malley" and "REN" stay whole letter runs. Matching them
    # case-insensitively would give "'M" + "alley" and "'RE" + "N".
    "O'Malley WE'REN'T",
]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=1) + "\n",
                    encoding="ascii", newline="\n")
    print(path)


def reference_tokenizer():
    source = MODEL_DIR / "tokenizer.json"
    if not source.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO_ID, "tokenizer.json", revision=REVISION,
                        local_dir=str(MODEL_DIR))
    target = OUT_DIR / "tokenizer.json"
    shutil.copyfile(source, target)
    print(target)
    return tokenizers.Tokenizer.from_file(str(target))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = reference_tokenizer()

    cases = []
    for text in FIXTURE_STRINGS:
        ids = tok.encode(text, add_special_tokens=False).ids
        decoded = tok.decode(ids, skip_special_tokens=False)
        cases.append({"text": text, "ids": ids, "decoded": decoded})
    write_json(OUT_DIR / "encode.json", {
        "revision": REVISION,
        "tokenizers_version": tokenizers.__version__,
        "cases": cases,
    })

    corpus = ROOT / CORPUS
    if corpus.exists():
        data = corpus.read_bytes()
        ids = tok.encode(data.decode("utf-8"), add_special_tokens=False).ids
        packed = struct.pack(f"<{len(ids)}i", *ids)
        write_json(OUT_DIR / "corpus.json", {
            "path": CORPUS,
            "bytes": len(data),
            "tokens": len(ids),
            "crc32": zlib.crc32(packed),
        })


if __name__ == "__main__":
    main()
