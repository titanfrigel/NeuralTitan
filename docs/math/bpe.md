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

Stage 1 uses a fourth class, the numbers of the Digits stage (`is_numeric`). It is defined by the same general categories, Nd, Nl and No, but it comes from a different Unicode table, as the next section explains.

### Where the classes come from

General categories change between Unicode versions: each version adds scripts, with their letters and digits. A character the tokenizer does not know as a letter splits a word in two, which changes the token ids. Exact reproduction therefore needs the reference's own Unicode version, and the reference uses two:

- the Digits stage calls Rust's `char::is_numeric`, from the Rust standard library the reference was built with;
- the GPT-2 pattern runs in the Oniguruma regex engine, which has its own Unicode tables.

Both are newer than Python's `unicodedata` (Unicode 15.1). For example, the Unicode 16 digits U+1CCF1 and U+16D71 and the letters U+11380 and U+10D50 are unknown to Python 3.13. Both are also different from each other: the digits U+11DE0–U+11DE9 are numbers for Rust but not for Oniguruma.

So `tools/gen_unicode_tables.py` does not read a Unicode database. It probes the reference itself, `tokenizers` 0.23.3 (the script refuses any other version), with every Unicode scalar value `c`:

| Class | Probe | Member when |
| --- | --- | --- |
| `is_numeric` | `Digits` on `"a" c "a"` | it gives 3 pieces: Digits isolated `c` |
| `is_letter` | GPT-2 pattern on `"a" c` | it gives 1 piece: `c` joined the letter run |
| `is_number` | GPT-2 pattern on `"1" c` | it gives 1 piece: `c` joined the number run |
| `is_whitespace` | GPT-2 pattern on `c "\n"` | it gives 1 piece: `\s+(?!\S)` took both |

The script checks each probe on characters whose class is known, that the three pattern classes are disjoint, that the probed `\s` equals the `White_Space` list above, and that every `\p{N}` number is also an `is_numeric` number. Against `unicodedata` 15.1, the probes add 4302 letters, 80 `\p{N}` numbers and 93 `is_numeric` numbers, remove nothing, and leave whitespace unchanged.

### Stage 1: digits

Every `is_numeric` code point becomes a piece of its own, and each run of code points between two of them becomes a piece. So `"ab123"` becomes `ab`, `1`, `2`, `3`. Numbers are then always encoded digit by digit, which gives the model a consistent view of them instead of arbitrary multi-digit tokens such as `"202"`.

### Stage 2: the GPT-2 split

Each piece from stage 1 is scanned left to right with the GPT-2 pattern:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

At each position the first alternative that matches is taken, the match becomes a piece, and the scan continues after it. The alternatives, in order:

1. `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`: English contractions, so `don't` becomes `don` and `'t`. They use the ASCII apostrophe and are case-sensitive: in `I'M`, `'M` matches none of them.
2. `␣?\p{L}+`: a run of letters, with at most one space (U+0020, written ␣) before it. Words carry their leading space, so `" the"` and `"the"` are different tokens, and decoding needs no rule about where spaces go.
3. `␣?\p{N}+`: the same for numbers. In SmolLM2 stage 1 has already made every number a piece of its own (every `\p{N}` number is an `is_numeric` number), so the pattern never sees a number next to anything else, not even a space.
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
| `\p{L}`, `\p{N}`, `\s`, the Digits-stage numbers | `is_letter`, `is_number`, `is_whitespace`, `is_numeric` in `src/tokenizer/unicode_tables.{hpp,cpp}`: a binary search over sorted, merged `{first, last}` ranges |
| The tables | Generated by `tools/gen_unicode_tables.py`, probed from `tokenizers` 0.23.3 (recorded as `kUnicodeSource`); the `White_Space` list is a constant in the script, checked against the probe |
| Both stages | `pretokenize` in `src/tokenizer/pretokenizer.cpp`; it decodes the UTF-8 once, rejecting invalid bytes with their byte offset, then returns views into the input |
| The pattern alternatives | `match_at`, a hand-written scanner with one block per alternative, in the pattern's order (no `std::regex`) |

## Byte-level alphabet

BPE in Hugging Face `tokenizers` works on strings, and its vocabulary file is JSON text. Raw bytes such as `0x00` or a lone `0xC3` cannot appear in either. GPT-2's answer, `bytes_to_unicode`, gives every byte value a printable stand-in code point, its **symbol**:

- bytes 33–126 (`!` to `~`), 161–172 (`¡` to `¬`) and 174–255 (`®` to `ÿ`) are printable already and map to the same code point;
- the other 68 bytes (0–32, 127–160 and 173), in increasing order, map to U+0100, U+0101, …, U+0143.

So the space `0x20`, the 33rd of the shifted bytes, becomes U+0100 + 32 = U+0120 `Ġ`, and the newline `0x0A` becomes U+010A `Ċ`. The map is one-to-one. A vocabulary entry is a string of symbols, and every token stands for a definite byte string: `"Ġthe"` is the four bytes of a space followed by `the`. A multi-byte UTF-8 character becomes several symbols: `é` is `C3 A9`, written `Ã©`.

## Encode

`encode(text)` runs four steps.

1. **Added tokens.** Special tokens such as `<|endoftext|>` are matched literally, before anything else. At each byte position the longest added token that matches is taken and emits its id. The text between matches is ordinary text. A partial token such as `<|endoftext` matches nothing and is encoded as ordinary text.
2. **Pre-tokenize** each span of ordinary text, as described above.
3. **Merge.** Each piece starts as one symbol per byte. While some adjacent pair of symbols has a merge, take the pair with the lowest rank (its index in the merge list), and replace all of its non-overlapping occurrences, from left to right, with the merged token. Repeat until no adjacent pair has a merge.
4. **Emit** the id of every remaining symbol.

Overlaps are resolved left to right: with only the merge `a a`, `aaa` becomes `aa`, `a`.

### Merge order

The reference merges one occurrence at a time. It keeps a priority queue of adjacent pairs ordered by rank, then by position. After each merge it adds the two new pairs that the merged token forms with its neighbours. Merging every occurrence of the lowest-ranked pair at once gives the same result only under a **precondition** on the merge list:

> 1. Every part of merge `i` that has more than one symbol is the result of some merge `j < i`.
> 2. No token is the result of two merges.

Together they give every multi-symbol token exactly one creator, and every merge that uses a token as a part ranks after that creator.

Under the precondition, suppose the reference has just merged an occurrence of the pair of rank `r` into the token `t`. Every new pair contains `t`. The merge of rank `r` is `t`'s only creator, so any merge that uses `t` as a part has a rank above `r`. So every new pair ranks above `r`. The queue therefore pops all remaining occurrences of rank `r`, from left to right, before anything else. An occurrence that overlaps a merged one has lost a symbol and is skipped. That is exactly what one step of our loop does.

Without either condition the two orders can differ. Both counterexamples below were checked against `tokenizers` 0.23.3.

**Condition 1.** Take the vocabulary `a`, `b`, `ab`, `aba` and the merges `ab a` (rank 0), then `a b` (rank 1). For `"abab"`:

- the reference merges the first `a b` into `ab`, which forms `ab a` of rank 0. That pair ranks below the second `a b`, so it is merged next. The result is `aba`, `b`;
- merging every `a b` at once gives `ab`, `ab`.

**Condition 2.** Take the vocabulary `bc`, `ab`, `abc`, `abca` and the merges `b c` (0), `a b` (1), `ab c` (2), `abc a` (3), `a bc` (4). Here `abc` has two creators, ranks 2 and 4. For `"abcabc"`:

- the reference merges both `b c` (rank 0), giving `a bc a bc`. Both `a b` pairs are now gone. It then merges the first `a bc` (rank 4) into `abc`, which forms `abc a` of rank 3. That pair ranks below the remaining `a bc`, so it is merged next. The result is `abca`, `bc`;
- merging every `a bc` at once gives `abc`, `abc`.

`from_json` and `from_parts` therefore reject:

- a merge with a multi-symbol part that no earlier merge creates, naming `model.merges[i]` and the part;
- a merge whose result an earlier merge already creates, naming `model.merges[i]`, the result and the earlier merge.

SmolLM2's 48900 merges meet both conditions: each multi-symbol part is created earlier, and no result repeats. NeuralTitan's own trainer must meet them as well, since `from_parts` checks them: every merge it learns has to add a new token.

For the same reason an added token may not be the symbol of a single byte. The reference would use the added token's id for that byte inside BPE, while NeuralTitan keeps added tokens out of BPE.

### Worked example: merging "hello"

Take the test tokenizer of `tests/tokenizer/bpe_test.cpp`, whose merges are `h e` (rank 0), `l l` (rank 1), `he ll` (rank 2) and `Ġ w` (rank 3). Encoding the piece `"hello"`:

| Step | Symbols | Candidate pairs and their ranks | Merge applied |
| --- | --- | --- | --- |
| 0 | `h e l l o` | `h e` 0, `l l` 1 | rank 0, `h e` |
| 1 | `he l l o` | `l l` 1 | rank 1, `l l` |
| 2 | `he ll o` | `he ll` 2 | rank 2, `he ll` |
| 3 | `hell o` | none | stop |

The result is the ids of `hell` and `o`. In `"hello world"`, the second piece `" world"` becomes `Ġw o r l d` through rank 3.

### Bytes without a symbol

A vocabulary need not hold all 256 single-byte symbols. SmolLM2's lacks 21 of them: `0x04`, `0x06`, `0x13`, `0x14`, `0x16`, `0x1D`, `0xC0`, `0xC1`, `0xF1`, `0xF2` and `0xF5`–`0xFF`. Its `unk_token` is null, so the reference silently drops such a byte: `"a\x04b"` encodes to `a`, `b`. NeuralTitan does the same, because:

- it reproduces the reference ids exactly;
- the model never saw those bytes in training, so they have no id to give it.

The byte is dropped when the piece is turned into symbols, before any merge. Its neighbours in the same piece therefore become adjacent and may merge: `"!\x04!"` gives the single token `!!`, exactly as with SmolLM2. Most such bytes are control characters, which the pre-tokenizer puts in a piece of their own. So `"he\x04llo"` is cut into the pieces `he`, `\x04` and `llo`, and gives `he`, `llo` rather than `hello`. A piece made only of such bytes yields no ids. Our own trainer always keeps all 256 symbols, so this case only arises with imported vocabularies.

## Decode

`decode(ids)` concatenates the tokens' byte strings:

- an ordinary token maps each of its symbols back to its byte;
- an added token yields its literal text, such as `<|endoftext|>`.

Every id is checked against `[0, vocab_size)` first, and an error names the id and its position. The result is raw bytes. A sequence of ids can end inside a multi-byte character, since one token may hold only the first byte of `é`. So the output is not guaranteed to be valid UTF-8, and callers that print it must handle an incomplete tail.

`decode(encode(text)) = text` holds for every valid UTF-8 text whose bytes all have a symbol: the merges only regroup symbols, and the added tokens and pieces cover the text in order. A dropped byte is lost.

## File format

The tokenizer is stored in Hugging Face's `tokenizer.json` layout, restricted to what SmolLM2 and our trainer use. `from_json` rejects anything else with an error naming the field path, such as `model.merges[3]` or `added_tokens[0].id`, so an unsupported file never loads as a silently different tokenizer.

| Field | Required value |
| --- | --- |
| `version` | `"1.0"` or absent |
| `normalizer`, `post_processor`, `truncation`, `padding` | `null` or absent |
| `pre_tokenizer` | `{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}`, compared field by field |
| `decoder` | `{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}`, compared field by field |
| `model.type` | `"BPE"` |
| `model.dropout`, `unk_token`, `continuing_subword_prefix`, `end_of_word_suffix` | `null` or absent |
| `model.fuse_unk`, `byte_fallback`, `ignore_merges` | `false` or absent |
| `model.vocab` | an object from text to id. The ids are integers, unique and dense over `[0, n)`, with 1 ≤ n < 2^31. Every entry that is not an added token is a non-empty string of byte-level symbols. |
| `model.merges` | each entry is `"a b"`, with exactly one space, or `["a", "b"]`. `a`, `b` and `a+b` are ordinary vocabulary entries, and no pair appears twice. A part of more than one symbol is the result of an earlier merge, and no two merges have the same result (see "Merge order"). The index is the rank. |
| `added_tokens[i]` | `content` is non-empty, is not the symbol of a single byte, and is in the vocabulary; `id` equals its vocabulary id. `special` is `true`; `normalized`, `lstrip`, `rstrip` and `single_word` are `false`. No content appears twice. |

Unknown keys are rejected at the top level, in `model` and in each added token. Ids are bounds-checked before anything is indexed or allocated.

`to_json` writes the same layout, with every field above present: `"version": "1.0"`, the reference pre-tokenizer and decoder, the vocabulary in id order, and merges as `"a b"` strings. An ordinary token never holds a raw space, which is the symbol `Ġ`, so `"a b"` is unambiguous. A tokenizer trained by NeuralTitan uses this layout too, with `<|endoftext|>` reserved at id 0.

### Where encode, decode and the file format live in the code

| Concept | Code |
| --- | --- |
| The byte-level map | `byte_to_symbol`, `symbol_to_byte`, `bytes_to_symbols`, `symbols_to_bytes` in `src/tokenizer/byte_level.{hpp,cpp}`; both tables are built at compile time |
| Validation of `tokenizer.json` | `BpeTokenizer::from_json` in `src/tokenizer/bpe_tokenizer.cpp`; it hands the checked parts to `from_parts` |
| The lookup tables | `ids_` (text → id), `byte_ids_` (byte → symbol id, or `kNoSymbol` for a byte that is dropped), `merge_of_` (`(left id, right id)` → `{rank, merged id}`) |
| Added-token split, pre-tokenization | `BpeTokenizer::encode` |
| The merge loop | `BpeTokenizer::encode_piece` |
| Decode | `BpeTokenizer::decode`, through `bytes_` (id → bytes) |
| Writing | `BpeTokenizer::to_json` |
