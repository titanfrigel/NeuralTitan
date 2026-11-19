cmake_minimum_required(VERSION 3.28)

project(NeuralTitan VERSION 0.1.0 LANGUAGES CXX)

# Checked before CUDA is enabled: nvcc's own compiler detection fails first
# with an opaque error when the host toolset is too new.
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
  message(FATAL_ERROR
    "MSVC ${MSVC_VERSION} is not supported by CUDA 12.6; use toolset 14.38 (scripts/dev-env.ps1)")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit 12.6 REQUIRED)

# Warnings are errors (ADR 0008): /W4 /WX for MSVC, and for nvcc every
# warning is an error with /W4 /WX forwarded to the host compiler.
if(MSVC)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:/W4;/WX;/permissive-;/utf-8>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings;-Xcompiler=/W4,/WX,/utf-8>")
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tests)
