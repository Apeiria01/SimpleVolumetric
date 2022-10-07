#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#define UPPER_ALIGN(A,B) ((UINT)(((A)+((B)-1))&~(B - 1)))

using namespace DirectX;

struct TexturedVertex
{
    XMFLOAT4 position;
    XMFLOAT2 texCoord;
};

struct Texture {
    XMFLOAT4 color;
};
void AllocateFrameBuffer(size_t width, size_t height);

void ReleaseFrameBuffer();

void CUDAWriteToTex(size_t width, size_t height,
    cudaSurfaceObject_t cudaDevVertptr, cudaStream_t streamToRun,
    float AnimTime, size_t currentFrame);