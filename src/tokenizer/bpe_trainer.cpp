#include "tokenizer/bpe_trainer.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "core/error.hpp"
#include "tokenizer/byte_level.hpp"
#include "tokenizer/pretokenizer.hpp"

namespace titan::tokenizer {

namespace {

constexpr std::string_view kEndOfText = "<|endoftext|>";
constexpr std::int32_t kEndOfTextId = 0;
// Byte b is id 1 + b; merge i creates id 257 + i.
constexpr std::int32_t kFirstByteId = 1;
constexpr std::int64_t kFirstMergeId = 257;

// A distinct pre-tokenized piece of the corpus: its current ids, which start
// as one id per byte, and the number of times it occurs.
struct Word {
    std::vector<std::int32_t> ids;
    std::int64_t freq;
};

// The words of `corpus`: <|endoftext|> is removed and splits the corpus, and
// each span between two of them is pre-tokenized. Sorted by their bytes, so
// the order does not depend on hashing.
std::vector<Word> count_words(std::string_view corpus) {
    std::map<std::string_view, std::int64_t> counts;
    std::size_t start = 0;
    while (true) {
        const std::size_t end = corpus.find(kEndOfText, start);
        const std::string_view span = corpus.substr(start, end - start);
        std::vector<std::string_view> pieces;
        try {
            pieces = pretokenize(span);
        } catch (const Error& e) {
            throw_error(__FILE__, __LINE__,
                        "bpe training: the corpus text starting at byte " +
                            std::to_string(start) + ": " + e.what());
        }
        for (const std::string_view piece : pieces) {
            counts[piece] += 1;
        }
        if (end == std::string_view::npos) break;
        start = end + kEndOfText.size();
    }

    std::vector<Word> words;
    words.reserve(counts.size());
    for (const auto& [piece, freq] : counts) {
        Word word{{}, freq};
        word.ids.reserve(piece.size());
        for (const char c : piece) {
            word.ids.push_back(kFirstByteId + static_cast<std::uint8_t>(c));
        }
        words.push_back(std::move(word));
    }
    return words;
}

// The tokens learned so far, as byte strings, and the set of those strings.
// <|endoftext|> counts as a token too.
class Tokens {
public:
    Tokens() {
        bytes_.emplace_back(kEndOfText);
        for (int b = 0; b < 256; ++b) {
            bytes_.emplace_back(1, static_cast<char>(b));
        }
        known_.insert(bytes_.begin(), bytes_.end());
    }

    // Records the merge of (left, right) and returns the new token's id.
    // Every merge creates a new token, so every token has exactly one
    // creator, as from_parts requires. Whole-word, left-to-right merging
    // guarantees it (docs/math/bpe.md, "Training"); a repeat is a bug.
    std::int32_t add(std::int32_t left, std::int32_t right) {
        std::string merged = text(left) + text(right);
        TITAN_CHECK(known_.insert(merged).second,
                    "bpe training: internal invariant broken (docs/math/"
                    "bpe.md, \"Merge order\" and \"Training\"): merge " +
                        std::to_string(bytes_.size() - kFirstMergeId) +
                        " recreates the existing token \"" +
                        bytes_to_symbols(merged) + "\"");
        bytes_.push_back(std::move(merged));
        return static_cast<std::int32_t>(bytes_.size() - 1);
    }

private:
    const std::string& text(std::int32_t id) const {
        return bytes_[static_cast<std::size_t>(id)];
    }

    std::vector<std::string> bytes_;  // id -> bytes
    std::unordered_set<std::string> known_;
};

// Replaces the non-overlapping occurrences of (left, right) in `ids`, left to
// right, with `merged`: the same rule as encode. Returns whether any did.
bool merge_word(std::vector<std::int32_t>& ids, std::int32_t left,
                std::int32_t right, std::int32_t merged) {
    std::size_t out = 0;
    bool changed = false;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i + 1 < ids.size() && ids[i] == left && ids[i + 1] == right) {
            ids[out++] = merged;
            ++i;
            changed = true;
        } else {
            ids[out++] = ids[i];
        }
    }
    ids.resize(out);
    return changed;
}

std::uint64_t pair_key(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left))
            << 32) |
           static_cast<std::uint32_t>(right);
}

std::int32_t key_left(std::uint64_t key) {
    return static_cast<std::int32_t>(key >> 32);
}

std::int32_t key_right(std::uint64_t key) {
    return static_cast<std::int32_t>(key & 0xFFFFFFFFu);
}

void check_max_merges(std::int64_t max_merges) {
    TITAN_CHECK(max_merges >= 0, "bpe training: max_merges " +
                                     std::to_string(max_merges) +
                                     " is negative");
}

// A max-heap entry. The best entry has the highest count, then the smallest
// left id, then the smallest right id: the heap orders (count, -left,
// -right).
struct HeapEntry {
    std::int64_t count;
    std::int32_t left;
    std::int32_t right;

    bool operator<(const HeapEntry& other) const {
        if (count != other.count) return count < other.count;
        if (left != other.left) return left > other.left;
        return right > other.right;
    }
};

}  // namespace

std::vector<BpeMerge> bpe_merges_naive(std::string_view corpus,
                                       std::int64_t max_merges) {
    check_max_merges(max_merges);
    std::vector<Word> words = count_words(corpus);
    Tokens tokens;
    std::vector<BpeMerge> merges;
    while (static_cast<std::int64_t>(merges.size()) < max_merges) {
        // Every adjacent pair of every word, weighted by the word's count.
        std::map<std::pair<std::int32_t, std::int32_t>, std::int64_t> counts;
        for (const Word& word : words) {
            for (std::size_t i = 0; i + 1 < word.ids.size(); ++i) {
                counts[{word.ids[i], word.ids[i + 1]}] += word.freq;
            }
        }
        // The map is in (left, right) order, so the first pair with the
        // highest count wins a tie.
        std::optional<BpeMerge> best;
        for (const auto& [pair, count] : counts) {
            if (!best || count > best->count) {
                best = BpeMerge{pair.first, pair.second, count};
            }
        }
        if (!best || best->count < 2) break;
        const std::int32_t merged = tokens.add(best->left, best->right);
        for (Word& word : words) {
            merge_word(word.ids, best->left, best->right, merged);
        }
        merges.push_back(*best);
    }
    return merges;
}

std::vector<BpeMerge> bpe_merges(std::string_view corpus,
                                 std::int64_t max_merges) {
    check_max_merges(max_merges);
    std::vector<Word> words = count_words(corpus);
    Tokens tokens;

    // counts[pair]: occurrences over all words, weighted by word count.
    // where[pair]: the indices of the words that contain the pair.
    std::unordered_map<std::uint64_t, std::int64_t> counts;
    std::unordered_map<std::uint64_t, std::unordered_set<std::int32_t>> where;
    const auto add_pairs = [&](std::int32_t index, std::int64_t sign) {
        const Word& word = words[static_cast<std::size_t>(index)];
        for (std::size_t i = 0; i + 1 < word.ids.size(); ++i) {
            const std::uint64_t key = pair_key(word.ids[i], word.ids[i + 1]);
            counts[key] += sign * word.freq;
            if (sign > 0) {
                where[key].insert(index);
            } else {
                where[key].erase(index);
            }
        }
    };
    for (std::size_t i = 0; i < words.size(); ++i) {
        add_pairs(static_cast<std::int32_t>(i), +1);
    }

    // Entries may be stale: a count only ever drops below its entry, except
    // for the pairs a merge touches, which get a fresh entry. A popped entry
    // whose count changed is re-pushed with the current count.
    std::priority_queue<HeapEntry> heap;
    for (const auto& [key, count] : counts) {
        heap.push({count, key_left(key), key_right(key)});
    }

    std::vector<BpeMerge> merges;
    while (static_cast<std::int64_t>(merges.size()) < max_merges &&
           !heap.empty()) {
        const HeapEntry top = heap.top();
        heap.pop();
        const std::uint64_t key = pair_key(top.left, top.right);
        const std::int64_t count = counts[key];
        if (count != top.count) {
            // A higher count always has its own entry, so only a lower one
            // needs re-pushing.
            if (count > 0 && count < top.count) {
                heap.push({count, top.left, top.right});
            }
            continue;
        }
        if (count < 2) break;

        const std::int32_t merged = tokens.add(top.left, top.right);
        merges.push_back({top.left, top.right, count});

        // Only the words that contain the pair change. Their pairs are
        // removed, the word is merged, and its new pairs are added back.
        const auto& containing = where[key];
        std::vector<std::int32_t> changed(containing.begin(), containing.end());
        std::ranges::sort(changed);
        std::unordered_set<std::uint64_t> touched;
        for (const std::int32_t index : changed) {
            add_pairs(index, -1);
            Word& word = words[static_cast<std::size_t>(index)];
            merge_word(word.ids, top.left, top.right, merged);
            add_pairs(index, +1);
            for (std::size_t i = 0; i + 1 < word.ids.size(); ++i) {
                touched.insert(pair_key(word.ids[i], word.ids[i + 1]));
            }
        }
        for (const std::uint64_t t : touched) {
            heap.push({counts[t], key_left(t), key_right(t)});
        }
    }
    return merges;
}

BpeTokenizer train_bpe(std::string_view corpus, std::int64_t vocab_size) {
    TITAN_CHECK(
        vocab_size >= kFirstMergeId &&
            vocab_size <= std::numeric_limits<std::int32_t>::max(),
        "train_bpe: vocab_size " + std::to_string(vocab_size) +
            " is outside [257, 2^31): it needs <|endoftext|> and 256 bytes");
    const std::vector<BpeMerge> learned =
        bpe_merges(corpus, vocab_size - kFirstMergeId);

    // Texts in the byte-level alphabet; a merge's text is its parts' texts.
    std::vector<std::string> vocab;
    vocab.reserve(static_cast<std::size_t>(kFirstMergeId) + learned.size());
    vocab.emplace_back(kEndOfText);
    for (int b = 0; b < 256; ++b) {
        const char byte = static_cast<char>(b);
        vocab.push_back(bytes_to_symbols(std::string_view(&byte, 1)));
    }
    std::vector<std::pair<std::int32_t, std::int32_t>> merges;
    merges.reserve(learned.size());
    for (const BpeMerge& m : learned) {
        vocab.push_back(vocab[static_cast<std::size_t>(m.left)] +
                        vocab[static_cast<std::size_t>(m.right)]);
        merges.emplace_back(m.left, m.right);
    }
    return BpeTokenizer::from_parts(std::move(vocab), std::move(merges),
                                    {kEndOfTextId});
}

}  // namespace titan::tokenizer
