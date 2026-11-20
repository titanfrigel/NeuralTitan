configure_file(
  version.hpp.in
  "${CMAKE_BINARY_DIR}/generated/core/version.hpp"
  @ONLY)

add_library(titan_core STATIC
  cuda_allocator.cpp
  cuda_check.cpp
  error.cpp
  parallel.cpp
  rng.cpp
  device.cpp
  storage.cpp
  tensor.cpp)

target_include_directories(titan_core
  PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${CMAKE_BINARY_DIR}/generated")

# Private: no public core header includes a CUDA header, so layers above core
# do not see the CUDA runtime. Executables that link titan_core still get
# cudart, because titan_core is a static library.
target_link_libraries(titan_core PRIVATE CUDA::cudart)
