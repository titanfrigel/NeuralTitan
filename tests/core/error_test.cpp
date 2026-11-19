#include "core/error.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"

#include <string>

TITAN_TEST(Error, CheckThrowsWithContext) {
    bool threw = false;
    try {
        TITAN_CHECK(1 == 2, "shapes [2,3] vs [3,2]");
    } catch (const titan::Error& e) {
        threw = true;
        const std::string what = e.what();
        TITAN_EXPECT(what.find("1 == 2") != std::string::npos);
        TITAN_EXPECT(what.find("shapes [2,3] vs [3,2]") != std::string::npos);
    }
    TITAN_EXPECT(threw);
}

TITAN_TEST(Error, CheckPassesSilently) { TITAN_CHECK(true, "x"); }

TITAN_TEST(Error, CudaCheckReportsErrorName) {
    TITAN_SKIP_IF_NO_CUDA()
    bool threw = false;
    try {
        TITAN_CUDA_CHECK(cudaSetDevice(9999));
    } catch (const titan::Error& e) {
        threw = true;
        const std::string what = e.what();
        TITAN_EXPECT(what.find("cudaErrorInvalidDevice") != std::string::npos);
    }
    TITAN_EXPECT(threw);
    TITAN_EXPECT(cudaGetLastError() == cudaSuccess);
}
