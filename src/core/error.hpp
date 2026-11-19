#pragma once

#include <stdexcept>
#include <string>

namespace titan {

// Base class of every exception the library throws.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Throws an Error whose message is "<file>:<line>: <message>".
[[noreturn]] void throw_error(const char* file, int line, std::string message);

}  // namespace titan

// Precondition check. `msg` is only evaluated when the check fails and may be
// a C string or a std::string. The result goes through a non-const variable so
// that a constant `cond` does not trigger MSVC's "conditional expression is
// constant" warning (C4127).
#define TITAN_CHECK(cond, msg)                                     \
    do {                                                           \
        bool titan_check_ok = static_cast<bool>(cond);             \
        if (!titan_check_ok) {                                     \
            ::titan::throw_error(                                  \
                __FILE__, __LINE__,                                \
                std::string("check failed: " #cond ": ") + (msg)); \
        }                                                          \
    } while (false)
