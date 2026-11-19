#include "harness/test.hpp"

#include <cstring>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    bool list_only = false;
    std::string_view filter;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--list") == 0) {
            list_only = true;
        } else {
            filter = argv[i];
        }
    }

    if (list_only) {
        titan::test::list_tests(filter, std::cout);
        return 0;
    }
    const titan::test::Summary summary =
        titan::test::run_tests(filter, std::cout);
    return summary.failed == 0 ? 0 : 1;
}
