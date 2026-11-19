// titan command-line entry point. M0 only offers --version and info.
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

#include "core/device.hpp"
#include "core/error.hpp"
#include "core/version.hpp"

namespace {

constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

int usage() {
    std::cerr << "usage: titan --version | info\n";
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
    if (argc != 2) return usage();
    const std::string_view command = argv[1];
    if (command == "--version") {
        std::cout << "titan " << titan::kVersion << '\n';
        return EXIT_SUCCESS;
    }
    if (command == "info") {
        print_info();
        return EXIT_SUCCESS;
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
    }
}
