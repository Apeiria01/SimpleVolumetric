#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <Eigen/Core>

using Eigen::Array4f;

#define UPPER_ALIGN(A,B) ((UINT)(((A)+((B)-1))&~(B - 1)))
void SimplePixelShader(size_t width, size_t height,
    cudaStream_t streamToRun,
    float AnimTime, size_t currentFrame, Array4f* buffer);