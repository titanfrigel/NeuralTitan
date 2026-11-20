#include "core/parallel.hpp"

#include <atomic>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/error.hpp"

namespace titan {

namespace {

int default_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

std::atomic<int> g_num_threads{default_threads()};

}  // namespace

int num_threads() { return g_num_threads.load(); }

void set_num_threads(int n) {
    TITAN_CHECK(n >= 1, "thread count must be >= 1, got " + std::to_string(n));
    g_num_threads.store(n);
}

void parallel_for(std::int64_t begin, std::int64_t end,
                  const std::function<void(std::int64_t, std::int64_t)>& body) {
    if (end <= begin) {
        return;
    }
    const std::int64_t count = end - begin;
    const std::int64_t threads = num_threads();
    if (count < 2 * threads) {
        body(begin, end);
        return;
    }

    std::exception_ptr first_error;
    std::mutex error_mutex;
    auto run_chunk = [&](std::int64_t b, std::int64_t e) {
        try {
            body(b, e);
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    };

    // Chunk t covers [begin + count * t / threads, begin + count * (t + 1) /
    // threads): sizes differ by at most one. The last chunk runs on the
    // calling thread.
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (std::int64_t t = 0; t + 1 < threads; ++t) {
        workers.emplace_back(run_chunk, begin + count * t / threads,
                             begin + count * (t + 1) / threads);
    }
    run_chunk(begin + count * (threads - 1) / threads, end);
    for (std::thread& w : workers) {
        w.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

}  // namespace titan
