#pragma once

#include <cstdint>
#include <functional>

namespace titan {

// Number of threads parallel_for uses. Defaults to the hardware concurrency,
// or 1 if that is unknown.
int num_threads();
// Throws if n < 1.
void set_num_threads(int n);

// Calls body(chunk_begin, chunk_end) on contiguous chunks that together cover
// [begin, end) exactly once, one chunk per thread. Runs inline on the calling
// thread when the range has fewer than 2 elements per thread. Waits for all
// chunks, then rethrows the first exception any of them threw.
void parallel_for(std::int64_t begin, std::int64_t end,
                  const std::function<void(std::int64_t, std::int64_t)>& body);

}  // namespace titan
