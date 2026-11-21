# The CUDA backend of ops. Only this library includes the CUDA runtime and
# cuBLAS headers among the ops libraries (docs/architecture/overview.md
# section 10). Its .hpp files are plain C++, so the dispatch in titan_ops,
# compiled by the host compiler, can include them.
add_library(titan_ops_cuda STATIC
  elementwise.cu
  reduce.cu
  softmax.cu
  norm.cu
  matmul.cu
  cublas_handle.cu)

target_include_directories(titan_ops_cuda PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(titan_ops_cuda
  PUBLIC titan_core
  PRIVATE CUDA::cudart CUDA::cublas)
