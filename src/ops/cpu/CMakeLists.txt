add_library(titan_ops_cpu STATIC
  elementwise.cpp
  reduce.cpp
  matmul.cpp
  softmax.cpp
  norm.cpp
  rope.cpp
  embedding.cpp
  cross_entropy.cpp
  adamw.cpp
  layout.cpp)

target_include_directories(titan_ops_cpu PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_ops_cpu PUBLIC titan_core)
