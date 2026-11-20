#include "nn/attention.hpp"

#include <cmath>
#include <utility>

#include "core/error.hpp"
#include "ops/elementwise.hpp"
#include "ops/layout.hpp"
#include "ops/matmul.hpp"
#include "ops/rope.hpp"
#include "ops/softmax.hpp"

namespace titan::nn {

namespace {

// Validates the head configuration and returns hd = dim / n_heads. Runs before
// the projections are built, so a bad configuration never divides by zero.
std::int64_t checked_head_dim(const std::string& name, std::int64_t dim,
                              std::int64_t n_heads, std::int64_t n_kv_heads,
                              PositionEncoding pos) {
    TITAN_CHECK(n_heads >= 1 && n_kv_heads >= 1,
                name + ": n_heads and n_kv_heads must be >= 1, got " +
                    std::to_string(n_heads) + " and " +
                    std::to_string(n_kv_heads));
    TITAN_CHECK(dim % n_heads == 0, name + ": dim " + std::to_string(dim) +
                                        " is not divisible by n_heads " +
                                        std::to_string(n_heads));
    TITAN_CHECK(n_heads % n_kv_heads == 0,
                name + ": n_heads " + std::to_string(n_heads) +
                    " is not divisible by n_kv_heads " +
                    std::to_string(n_kv_heads));
    const std::int64_t head_dim = dim / n_heads;
    TITAN_CHECK(pos != PositionEncoding::Rope || head_dim % 2 == 0,
                name + ": RoPE needs an even head dimension, got " +
                    std::to_string(head_dim));
    return head_dim;
}

// (B, T, H·hd) -> (B, H, T, hd).
Tensor split_heads(const Tensor& x, std::int64_t heads, std::int64_t head_dim) {
    return ops::swap_dims_1_2(x.reshape({x.dim(0), x.dim(1), heads, head_dim}));
}

// (B, H, T, hd) -> (B, T, H·hd), the inverse of split_heads.
Tensor merge_heads(const Tensor& x) {
    const Tensor t = ops::swap_dims_1_2(x);
    return t.reshape({t.dim(0), t.dim(1), t.dim(2) * t.dim(3)});
}

}  // namespace

Attention::Attention(std::string name, std::int64_t dim, std::int64_t n_heads,
                     std::int64_t n_kv_heads, PositionEncoding pos,
                     float rope_theta)
    : name_(name),
      dim_(dim),
      n_heads_(n_heads),
      n_kv_heads_(n_kv_heads),
      head_dim_(checked_head_dim(name, dim, n_heads, n_kv_heads, pos)),
      pos_(pos),
      rope_theta_(rope_theta),
      q_proj_(name + ".q_proj", dim, n_heads * head_dim_, false),
      k_proj_(name + ".k_proj", dim, n_kv_heads * head_dim_, false),
      v_proj_(name + ".v_proj", dim, n_kv_heads * head_dim_, false),
      o_proj_(name + ".o_proj", n_heads * head_dim_, dim, false) {}

Tensor Attention::forward(const Tensor& x, ForwardContext& ctx) {
    TITAN_CHECK(x.rank() == 3 && x.dim(2) == dim_,
                name_ + ": input must be (B, T, " + std::to_string(dim_) +
                    "), got " + to_string(x.shape()));
    const std::int64_t n_rep = n_heads_ / n_kv_heads_;
    const float score_scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    // Steps 1-2: project, then split into heads (B, H, T, hd).
    Tensor q = split_heads(q_proj_.forward(x, ctx), n_heads_, head_dim_);
    Tensor k = split_heads(k_proj_.forward(x, ctx), n_kv_heads_, head_dim_);
    const Tensor v =
        split_heads(v_proj_.forward(x, ctx), n_kv_heads_, head_dim_);
    // Step 3: rotate queries and keys by their absolute positions.
    if (pos_ == PositionEncoding::Rope) {
        q = ops::rope(q, ctx.pos_offset, rope_theta_);
        k = ops::rope(k, ctx.pos_offset, rope_theta_);
    }
    // Step 4: every query head h reads KV head h / n_rep.
    Tensor k_rep = ops::repeat_kv(k, n_rep);
    Tensor v_rep = ops::repeat_kv(v, n_rep);
    // Steps 5-6: scores = q·kᵀ / √hd, then the causal softmax per row.
    Tensor scores = ops::matmul(q, k_rep, false, true);
    ops::scale_(scores, score_scale);
    Tensor probs = ops::causal_softmax(scores, ctx.pos_offset);
    // Step 7: weighted sum of the values, heads merged back to (B, T, dim).
    const Tensor merged = merge_heads(ops::matmul(probs, v_rep));
    // Step 8: output projection.
    Tensor y = o_proj_.forward(merged, ctx);

    if (ctx.training) {
        q_ = std::move(q);
        k_rep_ = std::move(k_rep);
        v_rep_ = std::move(v_rep);
        probs_ = std::move(probs);
        pos_offset_ = ctx.pos_offset;
    } else {
        q_.reset();
        k_rep_.reset();
        v_rep_.reset();
        probs_.reset();
    }
    return y;
}

Tensor Attention::backward(const Tensor& dy) {
    require_training_forward(probs_.has_value(), name_);
    const std::int64_t n_rep = n_heads_ / n_kv_heads_;
    const float score_scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    // Step 8 and the head merge of step 7.
    const Tensor d_out = split_heads(o_proj_.backward(dy), n_heads_, head_dim_);
    // Step 7: out = probs·v_rep.
    const Tensor d_probs = ops::matmul(d_out, *v_rep_, false, true);
    const Tensor d_v_rep = ops::matmul(*probs_, d_out, true, false);
    // Steps 6-5: through the softmax, then scores = s·q·k_repᵀ.
    Tensor d_scores = ops::softmax_backward(*probs_, d_probs);
    ops::scale_(d_scores, score_scale);
    Tensor d_q = ops::matmul(d_scores, *k_rep_);
    const Tensor d_k_rep = ops::matmul(d_scores, *q_, true, false);
    // Step 4: each KV head collects the gradients of its n_rep copies.
    Tensor d_k = ops::repeat_kv_backward(d_k_rep, n_rep);
    const Tensor d_v = ops::repeat_kv_backward(d_v_rep, n_rep);
    // Step 3: RoPE is a rotation, so its backward is the inverse rotation.
    if (pos_ == PositionEncoding::Rope) {
        d_q = ops::rope_backward(d_q, pos_offset_, rope_theta_);
        d_k = ops::rope_backward(d_k, pos_offset_, rope_theta_);
    }
    // Steps 2-1: x feeds all three projections, so dx sums their gradients.
    Tensor dx = q_proj_.backward(merge_heads(d_q));
    ops::add_(dx, k_proj_.backward(merge_heads(d_k)));
    ops::add_(dx, v_proj_.backward(merge_heads(d_v)));
    return dx;
}

void Attention::parameters(std::vector<Parameter*>& out) {
    q_proj_.parameters(out);
    k_proj_.parameters(out);
    v_proj_.parameters(out);
    o_proj_.parameters(out);
}

}  // namespace titan::nn
