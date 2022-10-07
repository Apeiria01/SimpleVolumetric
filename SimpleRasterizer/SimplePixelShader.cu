#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "SimplePixelShader.h"
#include "GPUMemory.h"

__global__ void SimplePixelShaderDevice(unsigned int width,
    unsigned int height, float time, float4* buffer) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    float freq = 4.0f;
    if (y < height && x < width && abs(v - 0.5 * sinf(u * freq + time)) <= 0.001f) {
        //pixel = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
        buffer[y * width + x] = make_float4( 1.0f, 0.0f, 0.0f, 0.0f );
    }
    else {
        //buffer[y * width + x] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
}

void SimplePixelShader(size_t width, size_t height,
    cudaStream_t streamToRun,
    float AnimTime, size_t currentFrame, float4* buffer) {
    dim3 block(16, 16, 1);
    dim3 grid(UPPER_ALIGN(width, 16) / 16, UPPER_ALIGN(height, 16) / 16, 1);
    SimplePixelShaderDevice << <grid, block, 0, streamToRun >> > (width,
        height, AnimTime, buffer);
    getLastCudaError("circle_genTex_kernel execution failed.\n");
}