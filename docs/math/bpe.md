# Byte-level BPE tokenizer

The tokenizer that SmolLM2-135M uses: a GPT-2 style byte-level byte-pair encoding. NeuralTitan reproduces its token ids exactly. A tokenizer has no gradient, so this note replaces the forward and backward sections with the rules the code follows.

## What it computes

A language model reads and writes integer ids, not text. A tokenizer is the pair of maps between them:

- `encode`: UTF-8 text → a sequence of ids in `[0, vocab_size)`.
- `decode`: a sequence of ids → UTF-8 text, with `decode(encode(text)) = text`.

The character tokenizer of M1 to M3 gives every code point its own id. That keeps the vocabulary small but makes sequences long, and a model with a fixed context window then sees less text. Byte-pair encoding (BPE) instead learns a vocabulary of frequent multi-character pieces, so common words become one id. Byte-level BPE starts from the 256 byte values, so every string can be encoded and no character is ever unknown.

Encoding runs in two steps:

1. The **pre-tokenizer** cuts the text into pieces such as words with their leading space, single digits, punctuation runs and whitespace runs.
2. BPE encodes each piece on its own, and the ids of the pieces are concatenated.

### Why the text is pre-tokenized

BPE merges whichever adjacent pair is most frequent. Without a first split, it would happily learn tokens that straddle word boundaries, such as `"e t"` or `"g, the"`. Pieces like these depend on the words around them, so the same word would be encoded differently in different contexts. Cutting the text first means that:

- a merge never crosses a piece boundary, so a word gets the same ids wherever it appears;
- the vocabulary is spent on words and word parts rather than accidental neighbours;
- each piece is short, so encoding it is cheap, and repeated pieces can be cached.

## Pre-tokenizer

SmolLM2's pre-tokenizer is `Sequence[Digits(individual_digits=true), ByteLevel(add_prefix_space=false, use_regex=true)]`. It works on Unicode code points and returns pieces of the original text that concatenate back to it. (The byte-level step then maps each piece's bytes to printable symbols; that map is not part of the split.)

### Character classes

The pattern below uses three classes:

- `\p{L}`, letters: general categories Lu, Ll, Lt, Lm and Lo.
- `\p{N}`, numbers: general categories Nd (decimal digits), Nl (letter numbers such as `Ⅻ`) and No (other numbers such as `½`).
- `\s`, whitespace: the Unicode `White_Space` property, U+0009–000D, 0020, 0085, 00A0, 1680, 2000–200A, 2028, 2029, 202F, 205F and 3000.

Everything else, such as punctuation, symbols and emoji, is "other": `[^\s\p{L}\p{N}]`. Every code point falls in exactly one of the four.

### Stage 1: digits

Every `\p{N}` code point becomes a piece of its own, and each run of code points between two of them becomes a piece. So `"ab123"` becomes `ab`, `1`, `2`, `3`. Numbers are then always encoded digit by digit, which gives the model a consistent view of them instead of arbitrary multi-digit tokens such as `"202"`.

### Stage 2: the GPT-2 split

Each piece from stage 1 is scanned left to right with the GPT-2 pattern:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

At each position the first alternative that matches is taken, the match becomes a piece, and the scan continues after it. The alternatives, in order:

1. `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`: English contractions, so `don't` becomes `don` and `'t`. They use the ASCII apostrophe and are case-sensitive: in `I'M`, `'M` matches none of them.
2. `␣?\p{L}+`: a run of letters, with at most one space (U+0020, written ␣) before it. Words carry their leading space, so `" the"` and `"the"` are different tokens, and decoding needs no rule about where spaces go.
3. `␣?\p{N}+`: the same for numbers. After stage 1 a run holds a single number, so this yields a lone number, or a space and a number when the space is in the same stage-1 piece.
4. `␣?[^\s\p{L}\p{N}]+`: a run of "other" characters, with an optional leading space, such as `" '"` or `"?!"`.
5. `\s+(?!\S)`: a run of whitespace that is not followed by a non-space. The regex takes the longest run, then backtracks while the next character is a non-space. In practice:
   - at the end of the piece, the whole run matches;
   - before a non-space, the run minus its last character matches, so that last character, if it is a space, can attach to the next word through `␣?`;
   - a single whitespace character before a non-space does not match here.
6. `\s+`: any remaining whitespace run, which is the single whitespace character of the last case. A single space U+0020 before a non-space has already been claimed by alternatives 2 to 4, so this alternative catches any other single whitespace character before a non-space, such as a newline before a word.

The optional space in alternatives 2 to 4 is only U+0020. A no-break space U+00A0 before a word is whitespace of its own: `"a b"` splits into `a`, ` `, `b`.

Every code point matches at least one alternative (a letter matches 2, a number 3, an "other" 4, whitespace 6), so the scan always advances, and the pieces cover the text with no gaps and no empty pieces.

### Worked example

Take `"x  ' 12ab"`, which has two spaces after `x`.

Stage 1 isolates the digits: `"x  ' "`, `"1"`, `"2"`, `"ab"`.

Stage 2 scans `"x  ' "`:

| Position | Remaining | Alternative | Piece |
| --- | --- | --- | --- |
| 0 | `x␣␣'␣` | 2, a letter run | `x` |
| 1 | `␣␣'␣` | 5: the run `␣␣` is followed by `'`, so it gives back one space | `␣` |
| 2 | `␣'␣` | 4, a space then an "other" run | `␣'` |
| 4 | `␣` | 5: the run reaches the end of the piece | `␣` |

The single-character pieces `"1"`, `"2"` and the letter run `"ab"` pass through unchanged. The result is `x`, `␣`, `␣'`, `␣`, `1`, `2`, `ab`, which is what Hugging Face `tokenizers` gives (it shows the spaces as `Ġ` after the byte-level map).

### Where it lives in the code

| Concept | Code |
| --- | --- |
| `\p{L}`, `\p{N}`, `\s` | `is_letter`, `is_number`, `is_whitespace` in `src/tokenizer/unicode_tables.{hpp,cpp}`: a binary search over sorted, merged `{first, last}` ranges |
| The tables | Generated by `tools/gen_unicode_tables.py` from Python's `unicodedata` (Unicode 15.1.0, recorded as `kUnicodeVersion`); the `White_Space` list is a constant in the script |
| Both stages | `pretokenize` in `src/tokenizer/pretokenizer.cpp`; it decodes the UTF-8 once, rejecting invalid bytes with their byte offset, then returns views into the input |
| The pattern alternatives | `match_at`, a hand-written scanner with one block per alternative, in the pattern's order (no `std::regex`) |
