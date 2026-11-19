#pragma once

// Fixture for the layer check, scanned as text and never compiled.
// ops/cpu must not see its sibling backend or the CUDA runtime.
#include "ops/cuda/matmul.cuh"
#include <cuda_runtime.h>
