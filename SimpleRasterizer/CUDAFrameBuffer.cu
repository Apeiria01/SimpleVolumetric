#include "CudaFrameBuffer.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "surface_indirect_functions.h"

__global__ void framebufferToTex(cudaSurfaceObject_t tex, unsigned int width,
    unsigned int height, float4* buffer) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float4 pixel = {};
    if (y < height && x < width) {
        pixel = buffer[y * width + x];
    }
    else {

    }
    surf2Dwrite(pixel, tex, x * 16, y);
}

void CUDAFrameBuffer::Clear(cudaStream_t stream, UINT frameIndex) {
    cudaMemsetAsync(getRaw(frameIndex), 0, res.x() * res.y() * sizeof(Array4f), stream);
}

void CUDAFrameBuffer::WriteToTex(cudaStream_t stream, UINT frameIndex) {
    dim3 block(16, 16, 1);
    dim3 grid(res.x() / 16, res.y() / 16, 1);
    framebufferToTex <<<grid, block, 0, stream >>>(m_texture.getCudaTex(), res.x(), res.y(), getRaw(frameIndex));
}

float4* CUDAFrameBuffer::getRaw(int i)
{
    return frameBuffer[i % FrameCount].data();
}

CUDAFrameBuffer::CUDAFrameBuffer(size_t width, size_t height, D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle, D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle) :
    m_texture(width, height, cpuHandle, gpuHandle)
{
    res = Array2i(width, height);
    for (int i = 0; i < FrameCount; i++) {
        frameBuffer[i].allocate_memory(width * height * sizeof(float4));
    }
}

CUDAFrameBuffer::~CUDAFrameBuffer()
{
    for (int i = 0; i < FrameCount; i++) {
        frameBuffer[i].free_memory();
    }
}

