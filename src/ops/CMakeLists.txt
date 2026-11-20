add_subdirectory(cpu)

add_library(titan_ops STATIC
  elementwise.cpp
  reduce.cpp
  matmul.cpp
  softmax.cpp
  norm.cpp
  rope.cpp
  embedding.cpp
  cross_entropy.cpp
  adamw.cpp)

target_include_directories(titan_ops PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_ops PUBLIC titan_ops_cpu titan_core)
