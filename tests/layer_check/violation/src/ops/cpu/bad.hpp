#pragma once

// Fixture for the layer check, scanned as text and never compiled.
// ops/cpu must not see its sibling backend or the CUDA runtime, directly or
// through core/cuda_check.hpp.
#include "ops/cuda/matmul.cuh"
#include <cuda_runtime.h>
#include "core/cuda_check.hpp"
