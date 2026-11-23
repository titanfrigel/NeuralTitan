#include <cstdint>
#include <vector>

#include "core/error.hpp"
#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "harness/test.hpp"
#include "nn/embedding.hpp"
#include "nn/init.hpp"
#include "support/gradcheck.hpp"

using titan::Rng;
using titan::Tensor;
using titan::nn::Embedding;
using titan::nn::ForwardContext;
using titan::test::check_gradient;
using titan::test::random_tensor;

TITAN_TEST(EmbeddingModule, GradCheckWeight) {
    Rng rng(40);
    Embedding emb("tok", 7, 5);
    titan::nn::init_normal(emb.weight(), rng, 1.0f);
    // Repeated ids, so the scatter-add has rows that sum several gradients.
    const std::vector<std::int32_t> ids = {3, 0, 3, 6, 1, 3};
    const titan::Shape id_shape = {2, 3};
    const auto forward = [&]() {
        ForwardContext ctx;
        return emb.forward(ids, id_shape, ctx);
    };
    const Tensor r = random_tensor(forward().shape(), rng);

    std::vector<titan::nn::Parameter*> params;
    emb.parameters(params);
    titan::nn::zero_grad(params);
    emb.backward(r);
    TITAN_EXPECT_GRAD_OK(check_gradient(forward, emb.weight().value,
                                        emb.weight().grad.clone(), r));
}

TITAN_TEST(EmbeddingModule, BackwardWithoutTrainingForwardThrows) {
    Embedding emb("tok", 4, 3);
    ForwardContext ctx;
    ctx.training = false;
    const std::vector<std::int32_t> ids = {0, 1};
    const titan::Shape id_shape = {2};
    const Tensor y = emb.forward(ids, id_shape, ctx);
    TITAN_EXPECT_THROWS(emb.backward(y), titan::Error);
}
