#include "core/error.hpp"

namespace titan {

void throw_error(const char* file, int line, std::string message) {
    throw Error(std::string(file) + ":" + std::to_string(line) + ": " +
                message);
}

}  // namespace titan
