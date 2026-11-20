add_library(titan_tokenizer STATIC
  char_tokenizer.cpp)

target_include_directories(titan_tokenizer PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_tokenizer PUBLIC titan_io titan_core)
