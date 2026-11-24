add_library(titan_tokenizer STATIC
  bpe_tokenizer.cpp
  bpe_trainer.cpp
  byte_level.cpp
  char_tokenizer.cpp
  pretokenizer.cpp
  unicode_tables.cpp)

target_include_directories(titan_tokenizer PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_tokenizer PUBLIC titan_io titan_core)
