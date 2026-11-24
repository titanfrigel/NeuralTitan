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

## Training

Training learns the merge list from a corpus. It is the encoder run in reverse: each step finds the most frequent adjacent pair and turns it into a new token, until the vocabulary is full.

### Ids

- Id 0 is `<|endoftext|>`, the only added token.
- Byte `b` is id `1 + b`, so all 256 byte symbols are always present and no byte is ever dropped.
- Merge `i` creates id `257 + i`. Its text is the concatenation of its two parts' texts.

A tokenizer trained for `vocab_size` therefore learns at most `vocab_size − 257` merges, and `vocab_size` must be at least 257.

### Words

1. Every `<|endoftext|>` is removed from the corpus, which is split at it. The documents on either side never share a word, and the bytes of the marker never enter a merge.
2. Each remaining span is pre-tokenized, exactly as encode does it.
3. The distinct pieces are the **words**, each with its frequency `f(w)`, the number of times it occurs. Each word starts as one id per byte.

A merge never crosses a piece boundary in encode, so it never needs to in training: a word is the unit of work, and a corpus of a million bytes has only tens of thousands of distinct words.

### Each step

The **count** of a pair `(x, y)` is the number of adjacent positions where `x` is followed by `y`, summed over the words and weighted by their frequency:

```text
count(x, y) = Σ_w f(w) · #{ i : w[i] = x and w[i+1] = y }
```

Overlapping positions all count: `aaa` holds `(a, a)` twice.

1. Choose the pair with the highest count. A tie goes to the smallest `(left, right)`: the smallest left id, then the smallest right id. The order depends only on the ids, so training is deterministic.
2. Stop when that count is below 2, since a pair that occurs once saves nothing, or when `max_merges` merges have been made.
3. Create the new token, and in every word replace the pair's non-overlapping occurrences from left to right with it. This is the same rule as encode, so `aaa` becomes `aa a`. Encode then applies exactly the merges training applied, in the same order.

### Worked example: "aaab aaab ab"

The words are `aaab`, `Ġaaab` and `Ġab`, once each (`Ġ` is the leading space). The ids are `a` = 98, `b` = 99, `Ġ` = 33.

| Step | Words | Counts | Merge |
| --- | --- | --- | --- |
| 1 | `a a a b`, `Ġ a a a b`, `Ġ a b` | `(a, a)` 2 + 2 = 4; `(a, b)` 3; `(Ġ, a)` 2 | `(a, a)` → `aa`, id 257, count 4 |
| 2 | `aa a b`, `Ġ aa a b`, `Ġ a b` | `(a, b)` 3; `(aa, a)` 2; `(Ġ, aa)` 1; `(Ġ, a)` 1 | `(a, b)` → `ab`, id 258, count 3 |

In step 1, each `a a a` holds `(a, a)` twice, so the pair has 4. The merge replaces only the first of them, which leaves `aa a`. In step 2, `(aa, a)` has only 2, one per long word, and `(a, b)` wins with 3. `BpeTrain.HandExample` checks these two merges and their counts.

### Never recreate a token

Every merge must create a new token: `from_parts` and `from_json` reject a merge whose result an earlier merge already created (see "Merge order"). Training guarantees this by construction. **Invariant:** with whole-word, left-to-right merging, the chosen pair's concatenated text is never already a token. Proof: Suppose merge `k` were `(A, B)` with `A + B = T`, and an earlier merge `j < k` were `(C, D)` with `C + D = T` and `C ≠ A`.

- At step `k`, some word `w` holds `A` and `B` side by side. Tokens only ever grow, so before step `k` no token of `w` crosses either end of the span of `T`.
- Inside a span that no token crosses, the merges act as they would on the span alone. An occurrence that straddles an end of the span is either passed over, because a merge just before it took its first symbol, or merged into a token that crosses that end. So before step `k` the span of `T` in `w` is split exactly as `T` alone would be at the same step.
- At step `j`, some word `w'` holds `C` and `D` side by side, and by the same argument that is how `T` alone is split at step `j`.
- So at step `j` the span of `T` in `w` is also `C D`, and merge `j` turns it into the single token `T`. It can never again be `A B`, which contradicts step `k`.

Both trainers assert the invariant when they record a merge (`Tokens::add`): a repeated result would be a bug in the trainer, so it throws an error naming the merge index and the text instead of producing a file that `from_parts` rejects. `BpeTrain.MergesCreateDistinctTokens` checks that every created token is distinct and that the trained tokenizer loads from its own file.

### The incremental update

The naive trainer (`bpe_merges_naive`) recounts every pair of every word at every step. With `W` distinct words of total length `L`, a step costs `O(L + P log P)` for `P` distinct pairs, and `M` merges cost `O(M · L)` or more.

A merge changes only the words that contain its pair. `bpe_merges` keeps:

- `counts[pair]`, the weighted count;
- `where[pair]`, the set of words that contain the pair;
- a max-heap of entries `(count, −left, −right)`, so the top entry is the highest count with the smallest `(left, right)`.

When `(x, y)` is merged into `z`, each word in `where[(x, y)]` has all its pairs subtracted, is merged, and has its new pairs added back, which updates both `counts` and `where`. Every pair of a changed word gets a fresh heap entry with its new count.

Heap entries are not removed when a count changes, so an entry may be stale. A count only drops outside the pairs that get a fresh entry, so every pair has an entry at least as high as its count. When the top entry's count is not the current count:

- if the current count is lower, the entry is re-pushed with it;
- if it is higher, a fresher entry exists and the stale one is dropped.

The first entry that matches its pair's count is the true best pair. A step then costs `O(ℓ + e log H)`, where `ℓ` is the total length of the changed words, `e` the number of pairs in them and `H` the heap size. Frequent pairs sit in common words, but each merge touches only the words that hold it, not the whole corpus. `BpeTrain.MatchesNaive` checks that both trainers give the same merges and counts. On the full TinyShakespeare (1.1 MB), 3839 merges for a vocabulary of 4096 take about 0.25 s in a release build.

### Where training lives in the code

| Concept | Code |
| --- | --- |
| Words | `count_words` in `src/tokenizer/bpe_trainer.cpp` |
| Merging one word | `merge_word`, the same rule as `BpeTokenizer::encode_piece` |
| Token texts and the distinct-result invariant | `Tokens::add` |
| The naive trainer | `bpe_merges_naive` |
| Counts, `where`, the lazy heap | `bpe_merges`, `HeapEntry` |
| The tokenizer | `train_bpe`, which builds the vocabulary and calls `BpeTokenizer::from_parts` |
| The command | `titan tokenizer train --type bpe --vocab N` in `apps/titan/commands.cpp` |

## Decode

`decode(ids)` concatenates the tokens' byte strings:

- an ordinary token maps each of its symbols back to its byte;
- an added token yields its literal text, such as `<|endoftext|>`.

Every id is checked against `[0, vocab_size)` first, and an error names the id and its position. The result is raw bytes. A sequence of ids can end inside a multi-byte character, since one token may hold only the first byte of `é`. So the output is not guaranteed to be valid UTF-8, and callers that print it must handle an incomplete tail.

A streaming caller keeps the decoded bytes it has not printed yet and prints only `utf8_complete_prefix(bytes)` of them (`src/core/utf8.*`). That length leaves out a trailing lead byte with fewer continuation bytes than it announces, at most 3 bytes, when some valid sequence starts with them: `aæ` prints `a` and keeps `æ` until the next token completes `日`. Invalid bytes, including a lead no valid sequence starts with (`C0`, `C1`, `F5`…`FF`, or `E0 80`, `ED A0`, `F0 80`, `F4 90`, which would be overlong, a surrogate or above U+10FFFF), count as complete, so the caller never holds them back forever.

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
| The common interface | `tokenizer::Tokenizer` in `src/tokenizer/tokenizer.hpp`, which `CharTokenizer` and `BpeTokenizer` implement (`encode`, `decode`, `vocab_size`, `eos_id`, `to_json`); `load_tokenizer` picks the type from the document: `"type": "char"` or an object `model` |
| End of text | `BpeTokenizer::eos_id`, the id of the added end-of-text token; `Generator::generate` and `generate_naive` take it as `stop_id` and stop before emitting it |
