#include "ops/embedding.hpp"

#include <string>
#include <utility>

#include "ops/cpu/embedding.hpp"
#include "ops/detail.hpp"

namespace titan::ops {

namespace {

void check_table(const char* op, const Tensor& table) {
    TITAN_CHECK(table.rank() == 2, std::string("ops::") + op +
                                       ": table must have rank 2 (V, D), got " +
                                       to_string(table.shape()));
}

void check_ids(const char* op, const Tensor& table,
               std::span<const std::int32_t> ids) {
    const std::int64_t vocab = table.dim(0);
    for (std::size_t k = 0; k < ids.size(); ++k) {
        TITAN_CHECK(ids[k] >= 0 && ids[k] < vocab,
                    std::string("ops::") + op + ": id " +
                        std::to_string(ids[k]) + " at index " +
                        std::to_string(k) + " is out of range [0, " +
                        std::to_string(vocab) + ")");
    }
}

}  // namespace

Tensor embedding(const Tensor& table, std::span<const std::int32_t> ids,
                 Shape out_shape) {
    detail::require_cpu("embedding", {&table});
    check_table("embedding", table);
    TITAN_CHECK(out_shape.size() + 1 <= 4,
                "ops::embedding: out_shape " + to_string(out_shape) +
                    " plus the embedding dimension exceeds rank 4");
    // Stop multiplying as soon as the product exceeds the id count, so the
    // product cannot overflow.
    const auto id_count = static_cast<std::int64_t>(ids.size());
    std::int64_t count = 1;
    bool fits = true;
    for (std::int64_t d : out_shape) {
        if (d < 1 || d > id_count || count * d > id_count) {
            fits = false;
            break;
        }
        count *= d;
    }
    TITAN_CHECK(fits && count == id_count,
                "ops::embedding: out_shape " + to_string(out_shape) +
                    " does not match " + std::to_string(ids.size()) + " ids");
    check_ids("embedding", table, ids);
    out_shape.push_back(table.dim(1));
    return cpu::embedding(table, ids, std::move(out_shape));
}

void embedding_backward(Tensor& dtable_acc, std::span<const std::int32_t> ids,
                        const Tensor& dy) {
    detail::require_cpu("embedding_backward", {&dtable_acc, &dy});
    check_table("embedding_backward", dtable_acc);
    TITAN_CHECK(dy.dim(-1) == dtable_acc.dim(1) &&
                    dy.numel() == static_cast<std::int64_t>(ids.size()) *
                                      dtable_acc.dim(1),
                "ops::embedding_backward: dy shape " + to_string(dy.shape()) +
                    " does not match " + std::to_string(ids.size()) +
                    " ids and table " + to_string(dtable_acc.shape()));
    check_ids("embedding_backward", dtable_acc, ids);
    cpu::embedding_backward(dtable_acc, ids, dy);
}

}  // namespace titan::ops
