// CPU-vs-CUDA parity of the embedding lookup and its scatter-add backward.
// Both are exact: the forward copies rows, and the backward sums each element's
// repeated ids in double in ascending position order on both devices, then
// rounds once. So the results must match bit for bit (tolerance 0).

#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "ops/embedding.hpp"
#include "support/parity.hpp"

using titan::Device;
using titan::Rng;
using titan::Shape;
using titan::Tensor;
using titan::test::random_tensor;
namespace ops = titan::ops;

namespace {

// `count` ids drawn uniformly from [0, range), so a range far below the count
// repeats every id many times.
std::vector<std::int32_t> random_ids(std::int64_t count, std::int64_t range,
                                     std::uint64_t seed) {
    Rng rng(seed);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(count));
    for (auto& id : ids) id = static_cast<std::int32_t>(rng.randint(range));
    return ids;
}

struct Case {
    std::int64_t vocab;
    std::int64_t dim;
    Shape out_shape;
};

// A single element, a shape that is no multiple of 16 or 32 (37 x 129 table,
// 3 x 41 ids), and a result of 1024 * 1024 = 2^20 elements. In the last two
// cases there are more ids than vocabulary entries, so ids repeat.
const std::vector<Case>& cases() {
    static const std::vector<Case> c = {
        {1, 1, {1}}, {37, 129, {3, 41}}, {300, 1024, {2, 512}}};
    return c;
}

std::int64_t count_of(const Shape& s) {
    std::int64_t n = 1;
    for (std::int64_t d : s) n *= d;
    return n;
}

}  // namespace

TITAN_TEST(EmbeddingParity, Forward) {
    TITAN_REQUIRE_CUDA();
    for (const Case& c : cases()) {
        const Tensor table = random_tensor({c.vocab, c.dim}, 1);
        const auto ids = random_ids(count_of(c.out_shape), c.vocab, 2);
        const Tensor cpu = ops::embedding(table, ids, c.out_shape);
        const Tensor gpu =
            ops::embedding(table.to(Device::CUDA), ids, c.out_shape);
        TITAN_EXPECT(gpu.device() == Device::CUDA);
        TITAN_EXPECT(gpu.shape() == cpu.shape());
        TITAN_EXPECT_PARITY(cpu, gpu, 0.0, 0.0);
    }
}

// Repeated ids into an accumulator that already holds values: the scatter-add
// must sum the repeats and add to (not overwrite) what is there. Ids are drawn
// from the lower half of the vocabulary only, so the upper rows must come out
// unchanged and the lower ones repeat.
TITAN_TEST(EmbeddingParity, BackwardAccumulates) {
    TITAN_REQUIRE_CUDA();
    for (const Case& c : cases()) {
        const std::int64_t n = count_of(c.out_shape);
        const auto ids = random_ids(n, (c.vocab + 1) / 2, 3);
        Shape dy_shape = c.out_shape;
        dy_shape.push_back(c.dim);
        const Tensor dy = random_tensor(dy_shape, 4);
        const Tensor dy_gpu = dy.to(Device::CUDA);
        Tensor cpu_acc = random_tensor({c.vocab, c.dim}, 5);
        Tensor gpu_acc = cpu_acc.to(Device::CUDA);
        ops::embedding_backward(cpu_acc, ids, dy);
        ops::embedding_backward(gpu_acc, ids, dy_gpu);
        TITAN_EXPECT_PARITY(cpu_acc, gpu_acc, 0.0, 0.0);
        // A second call adds again.
        ops::embedding_backward(cpu_acc, ids, dy);
        ops::embedding_backward(gpu_acc, ids, dy_gpu);
        TITAN_EXPECT_PARITY(cpu_acc, gpu_acc, 0.0, 0.0);
    }
}

// Review focus 4: kernels do not validate, so an out-of-range id must be
// rejected on the host, with the CPU's message, before anything is uploaded.
// The accumulator is left untouched.
TITAN_TEST(EmbeddingParity, RejectsOutOfRangeIdOnCuda) {
    TITAN_REQUIRE_CUDA();
    const Tensor table = Tensor::zeros({4, 2}, Device::CUDA);
    for (std::int32_t bad : {-1, 4}) {
        const std::vector<std::int32_t> ids = {1, bad};
        const std::string expected =
            "id " + std::to_string(bad) + " at index 1 is out of range [0, 4)";
        try {
            ops::embedding(table, ids, {2});
            TITAN_EXPECT(false);
        } catch (const titan::Error& e) {
            const std::string msg = e.what();
            TITAN_EXPECT(msg.find("ops::embedding: " + expected) !=
                         std::string::npos);
        }
        Tensor dtable = Tensor::zeros({4, 2}, Device::CUDA);
        const Tensor dy = Tensor::zeros({2, 2}, Device::CUDA);
        try {
            ops::embedding_backward(dtable, ids, dy);
            TITAN_EXPECT(false);
        } catch (const titan::Error& e) {
            const std::string msg = e.what();
            TITAN_EXPECT(msg.find("ops::embedding_backward: " + expected) !=
                         std::string::npos);
        }
        for (float g : dtable.to(Device::CPU).to_vector()) {
            TITAN_EXPECT_EQ(g, 0.0f);
        }
    }
}
