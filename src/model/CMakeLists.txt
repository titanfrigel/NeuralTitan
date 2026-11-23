add_library(titan_model STATIC
  config.cpp
  model.cpp)

target_include_directories(titan_model PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_model PUBLIC titan_nn titan_io titan_ops titan_core)
