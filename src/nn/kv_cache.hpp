#pragma once

#include <cstdint>
#include <vector>

#include "core/device.hpp"
#include "core/tensor.hpp"

namespace titan::nn {

// The key/value head layout of one attention layer: how many KV heads it has
// and the size of each head.
struct KvLayerShape {
    std::int64_t n_kv_heads;
    std::int64_t head_dim;
};

// The keys and values of the positions already processed, one K and one V
// tensor per attention layer, so that generation runs each token through the
// model once. Layer i holds K and V of shape
// (batch, layers[i].n_kv_heads, max_len, layers[i].head_dim); rows
// 0..length()-1 are filled. Layers may differ in shape (heterogeneous blocks).
// Inference only: nothing here has a backward pass. See docs/math/kv-cache.md.
//
// The caller owns the length: a forward with ForwardContext::kv_cache set
// requires pos_offset == length() and writes rows pos_offset..pos_offset+T-1;
// afterwards the caller calls set_length(pos_offset + T).
class KvCache {
public:
    // Allocates zero-filled K and V tensors for every layer on `device`.
    // Throws unless layers is non-empty, batch >= 1, max_len >= 1 and every
    // layer has n_kv_heads >= 1 and head_dim >= 1.
    KvCache(std::vector<KvLayerShape> layers, std::int64_t batch,
            std::int64_t max_len, Device device);
    // Move-only: a copy would share the tensor storage but keep its own
    // length, so two copies would silently overwrite each other's rows.
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;
    KvCache(KvCache&&) = default;
    KvCache& operator=(KvCache&&) = default;

    // The cache of layer `layer`. Throws unless 0 <= layer < n_layers().
    Tensor& keys(std::int64_t layer);
    Tensor& values(std::int64_t layer);

    // Positions filled so far.
    std::int64_t length() const { return length_; }
    // Throws unless 0 <= length <= max_len().
    void set_length(std::int64_t length);
    // Forgets every position (length 0). The stale rows stay in memory but
    // are never read: every read covers only rows written since.
    void clear() { length_ = 0; }

    std::int64_t max_len() const { return max_len_; }
    std::int64_t batch() const { return batch_; }
    std::int64_t n_layers() const {
        return static_cast<std::int64_t>(layers_.size());
    }
    const KvLayerShape& layer_shape(std::int64_t layer) const;

private:
    std::vector<KvLayerShape> layers_;
    std::int64_t batch_;
    std::int64_t max_len_;
    std::int64_t length_ = 0;
    std::vector<Tensor> keys_;
    std::vector<Tensor> values_;

    void check_layer(std::int64_t layer) const;
};

}  // namespace titan::nn
