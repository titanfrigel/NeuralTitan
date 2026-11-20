#pragma once

#include <cmath>
#include <iomanip>
#include <iosfwd>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

namespace titan::test {

struct Summary {
    int passed;
    int failed;
    int skipped;
};

using TestFn = void (*)();

// Registers a test at static-initialisation time; used by TITAN_TEST.
struct Registrar {
    Registrar(const char* suite, const char* name, TestFn fn);
};

// Runs every registered test whose "Suite.Name" contains `filter`.
Summary run_tests(std::string_view filter, std::ostream& out);

// Prints the "Suite.Name" of every matching test, one per line.
void list_tests(std::string_view filter, std::ostream& out);

// Thrown by TITAN_SKIP; caught by the runner.
struct SkipException {
    std::string reason;
};

// Records a failed expectation against the running test and prints it.
void report_failure(const char* file, int line, const std::string& message);

namespace detail {

template <typename T>
std::string show(const T& value) {
    if constexpr (requires(std::ostream& os, const T& v) { os << v; }) {
        std::ostringstream os;
        // The default 6 significant digits can print two different floats
        // as the same number; max_digits10 always tells them apart.
        if constexpr (std::is_floating_point_v<T>) {
            os << std::setprecision(std::numeric_limits<T>::max_digits10);
        }
        os << value;
        return os.str();
    } else {
        return "<unprintable>";
    }
}

}  // namespace detail

}  // namespace titan::test

#define TITAN_TEST(Suite, Name)                                             \
    static void titan_test_##Suite##_##Name();                              \
    static const ::titan::test::Registrar titan_registrar_##Suite##_##Name( \
        #Suite, #Name, &titan_test_##Suite##_##Name);                       \
    static void titan_test_##Suite##_##Name()

#define TITAN_EXPECT(cond)                                            \
    do {                                                              \
        if (!(cond)) {                                                \
            ::titan::test::report_failure(__FILE__, __LINE__, #cond); \
        }                                                             \
    } while (false)

#define TITAN_EXPECT_EQ(a, b)                                         \
    do {                                                              \
        const auto& titan_lhs = (a);                                  \
        const auto& titan_rhs = (b);                                  \
        if (!(titan_lhs == titan_rhs)) {                              \
            ::titan::test::report_failure(                            \
                __FILE__, __LINE__,                                   \
                std::string(#a " == " #b " (") +                      \
                    ::titan::test::detail::show(titan_lhs) + " vs " + \
                    ::titan::test::detail::show(titan_rhs) + ")");    \
        }                                                             \
    } while (false)

#define TITAN_EXPECT_NEAR(a, b, tol)                                  \
    do {                                                              \
        const double titan_lhs = static_cast<double>(a);              \
        const double titan_rhs = static_cast<double>(b);              \
        const double titan_tol = static_cast<double>(tol);            \
        if (!(std::fabs(titan_lhs - titan_rhs) <= titan_tol)) {       \
            ::titan::test::report_failure(                            \
                __FILE__, __LINE__,                                   \
                std::string("|" #a " - " #b "| <= " #tol " (") +      \
                    ::titan::test::detail::show(titan_lhs) + " vs " + \
                    ::titan::test::detail::show(titan_rhs) + ")");    \
        }                                                             \
    } while (false)

#define TITAN_EXPECT_THROWS(expr, ExceptionType)                             \
    do {                                                                     \
        bool titan_threw_expected = false;                                   \
        try {                                                                \
            expr;                                                            \
        } catch (const ExceptionType&) {                                     \
            titan_threw_expected = true;                                     \
        } catch (...) {                                                      \
        }                                                                    \
        if (!titan_threw_expected) {                                         \
            ::titan::test::report_failure(                                   \
                __FILE__, __LINE__, #expr " did not throw " #ExceptionType); \
        }                                                                    \
    } while (false)

#define TITAN_SKIP(reason) \
    throw ::titan::test::SkipException { (reason) }
