add_library(titan_runtime STATIC
  data_loader.cpp
  sampler.cpp
  trainer.cpp)

target_include_directories(titan_runtime PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_runtime PUBLIC titan_model titan_optim titan_tokenizer titan_nn titan_io titan_ops titan_core)
