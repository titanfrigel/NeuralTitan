#include "harness/test.hpp"

#include <algorithm>
#include <exception>
#include <ostream>
#include <vector>

namespace titan::test {
namespace {

struct Entry {
    std::string suite;
    std::string name;
    TestFn fn;

    std::string full_name() const { return suite + "." + name; }
};

// Function-local static: registrars in other translation units run during
// static initialisation, so the registry must exist on first use.
std::vector<Entry>& registry() {
    static std::vector<Entry> entries;
    return entries;
}

struct Current {
    std::ostream* out = nullptr;
    std::string name;
    bool failed = false;
};

Current& current() {
    static Current state;
    return state;
}

std::vector<Entry> selected(std::string_view filter) {
    std::vector<Entry> result;
    for (const Entry& entry : registry()) {
        if (entry.full_name().find(filter) != std::string::npos) {
            result.push_back(entry);
        }
    }
    // Registration order across translation units is unspecified; sort so the
    // output is deterministic.
    std::sort(result.begin(), result.end(), [](const Entry& a, const Entry& b) {
        return a.full_name() < b.full_name();
    });
    return result;
}

}  // namespace

Registrar::Registrar(const char* suite, const char* name, TestFn fn) {
    registry().push_back(Entry{suite, name, fn});
}

void report_failure(const char* file, int line, const std::string& message) {
    Current& state = current();
    state.failed = true;
    if (state.out != nullptr) {
        *state.out << "FAIL " << state.name << " " << file << ":" << line << ": "
                   << message << "\n";
    }
}

void list_tests(std::string_view filter, std::ostream& out) {
    for (const Entry& entry : selected(filter)) {
        out << entry.full_name() << "\n";
    }
}

Summary run_tests(std::string_view filter, std::ostream& out) {
    Summary summary{0, 0, 0};
    for (const Entry& entry : selected(filter)) {
        Current& state = current();
        state.out = &out;
        state.name = entry.full_name();
        state.failed = false;

        bool skipped = false;
        std::string skip_reason;
        try {
            entry.fn();
        } catch (const SkipException& skip) {
            skipped = true;
            skip_reason = skip.reason;
        } catch (const std::exception& e) {
            report_failure("<unknown>", 0, std::string("unexpected exception: ") + e.what());
        } catch (...) {
            report_failure("<unknown>", 0, "unexpected non-standard exception");
        }

        if (state.failed) {
            out << "FAIL " << state.name << "\n";
            ++summary.failed;
        } else if (skipped) {
            out << "SKIP " << state.name << ": " << skip_reason << "\n";
            ++summary.skipped;
        } else {
            out << "PASS " << state.name << "\n";
            ++summary.passed;
        }
        state.out = nullptr;
    }
    out << summary.passed << " passed, " << summary.failed << " failed, "
        << summary.skipped << " skipped\n";
    return summary;
}

}  // namespace titan::test
