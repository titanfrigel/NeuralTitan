// titan command-line entry point: --version, info, tokenizer train, train,
// generate and bench matmul.
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include "commands.hpp"
#include "core/device.hpp"
#include "core/error.hpp"
#include "core/version.hpp"

namespace {

constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

int usage() {
    std::cerr << "usage: titan --version | info\n"
                 "       titan tokenizer train --type char --corpus <file> "
                 "--out <tok.json>\n"
                 "       titan train --config <cfg.json> --data <corpus.txt> "
                 "--out <dir> [--tokenizer <tok.json>] [--steps N] "
                 "[--stop-at N] [--device cpu|cuda] "
                 "[--gemm cublas|naive|tiled]\n"
                 "       titan train --resume <dir> [--stop-at N] "
                 "[--device cpu|cuda] [--gemm cublas|naive|tiled]\n"
                 "       titan generate --checkpoint <file> [--prompt <text>] "
                 "[--max-new N] [--temp T] [--top-k K] [--top-p P] [--seed S] "
                 "[--device cpu|cuda] [--no-cache]\n"
                 "       titan bench matmul [--sizes 256,512,1024,2048,4096] "
                 "[--repeat 10]\n";
    return kExitUsage;
}

void print_info() {
    std::cout << "CPU threads: " << std::thread::hardware_concurrency() << '\n';
    if (const auto device = titan::cuda_device_info()) {
        std::cout << "CUDA: " << device->name << ", "
                  << device->total_memory_bytes / (1024 * 1024)
                  << " MiB, compute " << device->compute_major << '.'
                  << device->compute_minor << '\n';
    } else {
        std::cout << "CUDA: not available\n";
    }
}

int run(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string_view command = argv[1];
    if (argc == 2 && command == "--version") {
        std::cout << "titan " << titan::kVersion << '\n';
        return EXIT_SUCCESS;
    }
    if (argc == 2 && command == "info") {
        print_info();
        return EXIT_SUCCESS;
    }
    const std::vector<std::string_view> args(argv + 2, argv + argc);
    try {
        if (command == "train") {
            titan::cli::train(args);
            return EXIT_SUCCESS;
        }
        if (command == "generate") {
            titan::cli::generate(args);
            return EXIT_SUCCESS;
        }
        if (command == "bench") {
            titan::cli::bench(args);
            return EXIT_SUCCESS;
        }
        if (command == "tokenizer" && !args.empty() && args[0] == "train") {
            titan::cli::tokenizer_train(std::span(args).subspan(1));
            return EXIT_SUCCESS;
        }
    } catch (const titan::cli::UsageError&) {
        return usage();
    }
    return usage();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const titan::Error& e) {
        std::cerr << "error: " << e.what() << '\n';
        return kExitError;
    } catch (const std::exception& e) {
        // Anything else that escapes, such as std::bad_alloc.
        std::cerr << "error: " << e.what() << '\n';
        return kExitError;
    }
}
