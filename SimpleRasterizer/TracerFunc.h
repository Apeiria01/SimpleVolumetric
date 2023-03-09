#pragma once
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "ShaderStructs.h"
#include <curand_kernel.h>

void CopyMatForR(const Eigen::Matrix4f* mat, cudaStream_t streamToRun);