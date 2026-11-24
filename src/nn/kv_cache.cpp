#include "nn/kv_cache.hpp"

#include <string>
#include <utility>

#include "core/error.hpp"

namespace titan::nn {

KvCache::KvCache(std::vector<KvLayerShape> layers, std::int64_t batch,
                 std::int64_t max_len, Device device)
    : layers_(std::move(layers)), batch_(batch), max_len_(max_len) {
    TITAN_CHECK(!layers_.empty(), "KvCache: needs at least one layer");
    TITAN_CHECK(batch_ >= 1 && max_len_ >= 1,
                "KvCache: batch and max_len must be >= 1, got " +
                    std::to_string(batch_) + " and " +
                    std::to_string(max_len_));
    keys_.reserve(layers_.size());
    values_.reserve(layers_.size());
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const KvLayerShape& s = layers_[i];
        TITAN_CHECK(s.n_kv_heads >= 1 && s.head_dim >= 1,
                    "KvCache: layer " + std::to_string(i) +
                        " needs n_kv_heads and head_dim >= 1, got " +
                        std::to_string(s.n_kv_heads) + " and " +
                        std::to_string(s.head_dim));
        const Shape shape{batch_, s.n_kv_heads, max_len_, s.head_dim};
        keys_.push_back(Tensor::zeros(shape, device));
        values_.push_back(Tensor::zeros(shape, device));
    }
}

void KvCache::check_layer(std::int64_t layer) const {
    TITAN_CHECK(layer >= 0 && layer < n_layers(),
                "KvCache: layer " + std::to_string(layer) +
                    " is out of range for " + std::to_string(n_layers()) +
                    " layers");
}

Tensor& KvCache::keys(std::int64_t layer) {
    check_layer(layer);
    return keys_[static_cast<std::size_t>(layer)];
}

Tensor& KvCache::values(std::int64_t layer) {
    check_layer(layer);
    return values_[static_cast<std::size_t>(layer)];
}

const KvLayerShape& KvCache::layer_shape(std::int64_t layer) const {
    check_layer(layer);
    return layers_[static_cast<std::size_t>(layer)];
}

void KvCache::set_length(std::int64_t length) {
    TITAN_CHECK(length >= 0 && length <= max_len_,
                "KvCache: length " + std::to_string(length) +
                    " is outside 0.." + std::to_string(max_len_));
    length_ = length;
}

}  // namespace titan::nn
