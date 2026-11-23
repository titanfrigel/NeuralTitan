# Sampling

## What it computes

Turns the logits of the last position, a vector `z` of length `V` (the vocabulary size), into one token id. Generation calls it once per new token. The controls are the temperature `T`, `top_k` and `top_p`. This is inference only: there is no backward pass.

## Forward

Temperature 0 is greedy: the result is `argmax_i z_i`, the lowest index among ties, and the random generator is not touched.

Otherwise, four steps, in this order:

1. Temperature softmax, in `double`: `w_i = exp((z_i − z_max) / T)`, `p_i = w_i / Σ_j w_j`. Subtracting `z_max` does not change `p` and keeps every exponent at or below 0 (see [Softmax](softmax.md)).
2. Top-k, when `top_k > 0`: keep the `top_k` tokens with the largest `p`. Ties at the cutoff keep the lower index.
3. Top-p, when `top_p < 1`: sort the kept tokens by `p` descending (lower index first on ties) and keep the shortest prefix whose probabilities sum to at least `top_p`. The sums use the distribution renormalized over the top-k survivors (the Hugging Face convention); with `top_k = 0` that is the step-1 `p`. The first token always survives, so the set is never empty.
4. Renormalize over the kept tokens and draw by inverse CDF with one `rng.uniform()`, walking the kept tokens in index order.

### Temperature

Dividing the logits by `T` before the softmax sharpens the distribution when `T < 1` and flattens it when `T > 1`. As `T → 0` all the mass goes to the argmax; as `T → ∞` it tends to uniform. Worked example with `z = (2, 1, 0)`:

| `T` | scaled logits | `p` |
| --- | --- | --- |
| 0.5 | (4, 2, 0) | (0.867, 0.117, 0.016) |
| 1 | (2, 1, 0) | (0.665, 0.245, 0.090) |
| 2 | (1, 0.5, 0) | (0.506, 0.307, 0.186) |

At `T = 1` the weights are `e² = 7.389`, `e = 2.718` and `1`, which sum to `11.107`.

### Top-k and top-p

Same example at `T = 1`, `p = (0.665, 0.245, 0.090)`:

- `top_k = 2` keeps tokens 0 and 1. Renormalized, `p = (0.665, 0.245) / 0.910 = (0.731, 0.269)` and token 2 is never drawn.
- `top_p = 0.9` needs the prefix sums `0.665` (below 0.9) and `0.910` (at least 0.9), so it keeps tokens 0 and 1, the same set. `top_p = 0.6` stops after the first token, since `0.665 ≥ 0.6`, and keeps token 0 alone.
- With both, top-k restricts the candidates first, the survivors are renormalized, and top-p then trims that set. Take `p = (0.5, 0.3, 0.15, 0.05)`, `top_k = 3`, `top_p = 0.83`: top-k keeps tokens 0 to 2 with mass `0.95`, renormalized to `(0.526, 0.316, 0.158)`; the prefix sums `0.526` and `0.842` reach `0.83` after two tokens, so only tokens 0 and 1 survive. Against the unrenormalized `p` the sums would be `0.5`, `0.8`, `0.95` and token 2 would stay.

Top-k cuts a fixed number of tokens whatever the shape of the distribution; top-p adapts, keeping few tokens when the model is confident and many when it is not.

### Why inverse CDF with one uniform is exact

Let the kept weights, in index order, be `w_1 … w_m` with total `S`, and let `c_i = w_1 + … + w_i`. Draw `u` uniform in `[0, 1)` and return the first `i` with `u·S < c_i`. The values `u·S` fall in `[c_{i−1}, c_i)` with probability `(c_i − c_{i−1}) / S = w_i / S`, which is exactly the renormalized probability of token `i`. One uniform is enough, and no token is rejected and redrawn. Scaling `u` by `S` is the same as renormalizing the weights, so dropped tokens simply get weight 0 and are skipped by the walk. `Rng::uniform()` is on a 24-bit grid, so probabilities are resolved to about `6·10⁻⁸`.

### Order of operations and RNG contract

- Greedy draws nothing. Every other call draws exactly one `rng.uniform()`, whatever `top_k` and `top_p` are, so a run consumes one number per generated token and a seed reproduces it.
- With `top_k = 0` and `top_p = 1` the result is bit-identical to the M1 sampler (`sample_token`), whose end-of-run sample in `titan train` depends on that. `Sampler.MatchesM1SampleToken` pins 200 draws recorded from the M1 code.
- Top-p is applied to the top-k-renormalized distribution, so the two filters compose in the order top-k, then top-p.
- The kept set depends only on the logits and the config, never on the random draw.

## Shapes

- `logits`: one position, a non-empty span of `V` floats.
- Result: one id in `[0, V)`.
- `validate(config)` throws, naming the field, for a temperature that is negative or non-finite, `top_k < 0`, or `top_p` outside `(0, 1]`. `top_k ≥ V` keeps everything.

## Mapping to code

- `runtime::SamplerConfig`, `runtime::validate`, `runtime::sample` and `runtime::sample_token` (the M1 entry point, now `sample(logits, {T, 0, 1}, rng)`): `src/runtime/sampler.hpp`, `src/runtime/sampler.cpp`. All probability arithmetic is in `double`.
- Tests: `tests/runtime/sampler_test.cpp`. Greedy ignores the generator, the M1 fixture, top-k never picks outside its set (and ties keep the lower index), top-p keeps the minimal set with the expected renormalized frequencies, top-p after top-k works on the renormalized survivors, frequencies match the softmax, and bad configs are rejected. The statistical tests use fixed seeds and fixed tolerances, so they are deterministic.
