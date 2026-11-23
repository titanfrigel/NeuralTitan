add_library(titan_optim STATIC
  adamw.cpp
  clip.cpp
  schedule.cpp)

target_include_directories(titan_optim PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_optim PUBLIC titan_nn titan_ops titan_core)
