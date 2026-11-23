#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"
#include "ops/embedding.hpp"

using titan::Device;
using titan::Tensor;
namespace ops = titan::ops;

namespace {

Tensor make(std::vector<float> v, titan::Shape shape) {
    return Tensor::from_vector(v, std::move(shape));
}

}  // namespace

TITAN_TEST(Embedding, GatherRows) {
    const Tensor table = make({0, 1, 10, 11, 20, 21, 30, 31}, {4, 2});
    const std::vector<std::int32_t> ids = {3, 0, 2, 2, 1, 3};
    const Tensor y = ops::embedding(table, ids, {2, 3});
    TITAN_EXPECT(y.shape() == titan::Shape({2, 3, 2}));
    TITAN_EXPECT(y.to_vector() == std::vector<float>({30, 31, 0, 1, 20, 21, 20,
                                                      21, 10, 11, 30, 31}));
}

TITAN_TEST(Embedding, ScatterAddRepeatedIds) {
    Tensor dtable = Tensor::zeros({4, 2}, Device::CPU);
    const std::vector<std::int32_t> ids = {2, 0, 2};
    const Tensor dy = make({1, 2, 3, 4, 5, 6}, {1, 3, 2});
    ops::embedding_backward(dtable, ids, dy);
    // Row 2 receives dy rows 0 and 2; row 0 receives dy row 1.
    TITAN_EXPECT(dtable.to_vector() ==
                 std::vector<float>({3, 4, 0, 0, 6, 8, 0, 0}));
    // The scatter accumulates into what is already there.
    ops::embedding_backward(dtable, ids, dy);
    TITAN_EXPECT(dtable.to_vector() ==
                 std::vector<float>({6, 8, 0, 0, 12, 16, 0, 0}));
}

TITAN_TEST(Embedding, OutOfRangeIdThrows) {
    const Tensor table = Tensor::zeros({4, 2}, Device::CPU);
    for (std::int32_t bad : {-1, 4}) {
        const std::vector<std::int32_t> ids = {1, bad};
        try {
            ops::embedding(table, ids, {2});
            TITAN_EXPECT(false);
        } catch (const titan::Error& e) {
            const std::string msg = e.what();
            TITAN_EXPECT(msg.find("id " + std::to_string(bad) +
                                  " at index 1") != std::string::npos);
        }
        Tensor dtable = Tensor::zeros({4, 2}, Device::CPU);
        const Tensor dy = Tensor::zeros({2, 2}, Device::CPU);
        TITAN_EXPECT_THROWS(ops::embedding_backward(dtable, ids, dy),
                            titan::Error);
    }
}

TITAN_TEST(Embedding, InvalidShapesThrow) {
    const Tensor table = Tensor::zeros({4, 2}, Device::CPU);
    const std::vector<std::int32_t> ids = {0, 1, 2, 3};
    TITAN_EXPECT_THROWS(ops::embedding(table, ids, {3}), titan::Error);
    TITAN_EXPECT_THROWS(ops::embedding(table, ids, {2, 3}), titan::Error);
    TITAN_EXPECT_THROWS(ops::embedding(table, ids, {1, 1, 2, 2}), titan::Error);
    TITAN_EXPECT_THROWS(
        ops::embedding(Tensor::zeros({4}, Device::CPU), ids, {4}),
        titan::Error);
    Tensor dtable = Tensor::zeros({4, 2}, Device::CPU);
    TITAN_EXPECT_THROWS(ops::embedding_backward(
                            dtable, ids, Tensor::zeros({4, 3}, Device::CPU)),
                        titan::Error);
    TITAN_EXPECT_THROWS(ops::embedding_backward(
                            dtable, ids, Tensor::zeros({3, 2}, Device::CPU)),
                        titan::Error);
}

TITAN_TEST(Embedding, CudaTensorThrowsM2Message) {
    TITAN_SKIP_IF_NO_CUDA();
    const Tensor table = Tensor::zeros({4, 2}, Device::CUDA);
    const std::vector<std::int32_t> ids = {0};
    try {
        ops::embedding(table, ids, {1});
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(std::string(e.what()).find(
                         "ops::embedding: CUDA implementation arrives in M2") !=
                     std::string::npos);
    }
}
