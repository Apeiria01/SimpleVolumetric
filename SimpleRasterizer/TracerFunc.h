#pragma once
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "ShaderStructs.h"
#include <curand_kernel.h>

void CopyMatForR(const Eigen::Matrix4f* mat, cudaStream_t streamToRun);
void CopySceneData(const Sphere* sph, UINT size, cudaStream_t streamToRun);
void CopyMaterialData(const Material* m, UINT size, cudaStream_t streamToRun);

void TracerPixelShader(unsigned int width,
    unsigned int height, Array4f* buffer,
    cudaStream_t streamToRun, const UINT maxSamplePerPixel, size_t frame);