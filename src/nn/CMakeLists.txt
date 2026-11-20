add_library(titan_nn STATIC
  module.cpp
  init.cpp
  linear.cpp
  embedding.cpp
  norm.cpp
  mlp.cpp
  attention.cpp
  block.cpp
  lm_head.cpp
  loss.cpp)

target_include_directories(titan_nn PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_nn PUBLIC titan_ops titan_core)
