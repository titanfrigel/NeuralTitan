configure_file(
  version.hpp.in
  "${CMAKE_BINARY_DIR}/generated/core/version.hpp"
  @ONLY)

add_library(titan_core STATIC version.cpp)

target_include_directories(titan_core
  PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${CMAKE_BINARY_DIR}/generated")

target_link_libraries(titan_core PUBLIC CUDA::cudart)
