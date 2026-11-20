add_library(titan_io STATIC
  json.cpp
  file.cpp
  token_file.cpp)

target_include_directories(titan_io PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_io PUBLIC titan_core)
