# Enforces the layer include rules of docs/architecture/overview.md section 10.
#
# Usage: cmake -DSRC_DIR=<dir> -P scripts/check_layers.cmake
#
# The layers form a one-way stack: core at the bottom, cli at the top. A lower
# layer that includes a higher one creates a cycle, drags in code that could
# not be built or tested without it, and makes a backend (ops/cpu, ops/cuda)
# depend on its sibling. CMake target_link_libraries already rejects forbidden
# calls at link time; this script also catches a forbidden #include early, with
# a file and line, before anything is built. It only reads text, so it needs no
# compiler and also scans files that no target compiles.
#
# Exit code 0 prints "layer check: OK (<n> files)". Otherwise every violation
# is listed as "<file>:<line>: <layer> may not include <target>".

cmake_minimum_required(VERSION 3.28)

if(NOT SRC_DIR)
  message(FATAL_ERROR "usage: cmake -DSRC_DIR=<dir> -P check_layers.cmake")
endif()
cmake_path(ABSOLUTE_PATH SRC_DIR NORMALIZE OUTPUT_VARIABLE SRC_DIR)

# The rules, copied from the table in the architecture overview: for each layer
# the other layers it may include. A layer may always include itself. The key
# has "/" written as "_" because CMake variable names are easier that way.
set(ALLOW_core "")
set(ALLOW_io "core")
set(ALLOW_tokenizer "io;core")
set(ALLOW_ops_cpu "core")
set(ALLOW_ops_cuda "core")
set(ALLOW_ops "ops/cpu;ops/cuda;core")
set(ALLOW_nn "ops;core")
set(ALLOW_optim "nn;ops;core")
set(ALLOW_model "nn;io;ops;core")
set(ALLOW_runtime "model;optim;tokenizer;nn;io;ops;core")
set(ALLOW_cli "runtime;model;optim;tokenizer;nn;io;ops;core")
set(ALL_LAYERS
  core io tokenizer ops/cpu ops/cuda ops nn optim model runtime cli)

# System headers that only some layers may include: the CUDA runtime headers
# stay in core and ops/cuda so the rest of the code reaches the GPU through
# core and ops, and cuBLAS stays in ops/cuda alone.
set(CUDA_RUNTIME_HEADERS cuda_runtime.h cuda.h cuda_runtime_api.h)
set(CUDA_RUNTIME_LAYERS core ops/cuda)
set(CUBLAS_HEADERS cublas_v2.h)
set(CUBLAS_LAYERS ops/cuda)

# Layer of a path relative to SRC_DIR: the first component, except that ops/cpu
# and ops/cuda are layers of their own. Empty when the path is in no layer.
function(layer_of rel_path out_var)
  if(rel_path MATCHES "^ops/(cpu|cuda)/")
    set(${out_var} "ops/${CMAKE_MATCH_1}" PARENT_SCOPE)
    return()
  endif()
  if(rel_path MATCHES "^([^/]+)/")
    set(first_component "${CMAKE_MATCH_1}")
    if(first_component IN_LIST ALL_LAYERS)
      set(${out_var} "${first_component}" PARENT_SCOPE)
      return()
    endif()
  endif()
  set(${out_var} "" PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE sources RELATIVE "${SRC_DIR}"
  "${SRC_DIR}/*.hpp" "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.cu" "${SRC_DIR}/*.cuh")
list(SORT sources)

set(violations "")
set(scanned 0)
foreach(file IN LISTS sources)
  layer_of("${file}" layer)
  if(layer STREQUAL "")
    continue()
  endif()
  math(EXPR scanned "${scanned} + 1")
  string(REPLACE "/" "_" layer_key "${layer}")

  # Read the whole file and turn it into a list with one element per line.
  # Brackets and semicolons are blanked first because they would split or merge
  # list elements; include lines contain neither.
  file(READ "${SRC_DIR}/${file}" content)
  string(REGEX REPLACE "[][;]" "_" content "${content}")
  string(REPLACE "\r" "" content "${content}")
  string(REPLACE "\n" ";" lines "${content}")

  set(line_number 0)
  foreach(line IN LISTS lines)
    math(EXPR line_number "${line_number} + 1")
    if(NOT line MATCHES "^[ \t]*#[ \t]*include[ \t]*([<\"])([^>\"]+)[>\"]")
      continue()
    endif()
    set(quote "${CMAKE_MATCH_1}")
    set(target "${CMAKE_MATCH_2}")

    set(forbidden FALSE)
    if(target IN_LIST CUDA_RUNTIME_HEADERS)
      if(NOT layer IN_LIST CUDA_RUNTIME_LAYERS)
        set(forbidden TRUE)
      endif()
    elseif(target IN_LIST CUBLAS_HEADERS)
      if(NOT layer IN_LIST CUBLAS_LAYERS)
        set(forbidden TRUE)
      endif()
    elseif(quote STREQUAL "\"")
      # A project include: forbidden when it points into a layer that this
      # layer may not use. Paths in no layer (a sibling header) are fine.
      layer_of("${target}" target_layer)
      if(NOT target_layer STREQUAL "" AND NOT target_layer STREQUAL layer
          AND NOT target_layer IN_LIST ALLOW_${layer_key})
        set(forbidden TRUE)
        set(target "${target_layer}")
      endif()
    endif()

    if(forbidden)
      list(APPEND violations "${file}:${line_number}: ${layer} may not include ${target}")
    endif()
  endforeach()
endforeach()

if(violations)
  list(JOIN violations "\n" report)
  message(FATAL_ERROR "layer check failed:\n${report}")
endif()
message(STATUS "layer check: OK (${scanned} files)")
