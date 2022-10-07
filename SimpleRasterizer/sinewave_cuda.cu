
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "ShaderStructs.h"
#include "surface_functions.h"
#include "GPUMemory.h"
#include <Eigen/Core>


GPUMemory<float4> frameBuffer[3];

__global__ void framebuffer_to_tex(cudaSurfaceObject_t tex, unsigned int width,
    unsigned int height, float4* buffer) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float4 pixel = {};
    if (y < height && x < width) {
        pixel = { buffer[y * width + x]};
    }
    else {
        //pixel = make_float4(w * 5.0f, 1.0f, 0.7f, 1.0f);
    }
    surf2Dwrite(pixel, tex, x * 16, y);
}

__global__ void write_to_tex_kernel(unsigned int width,
    unsigned int height, float time, float4* buffer) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    // calculate simple sine wave pattern
    float freq = 4.0f;
    //float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
    //float4 pixel{};
    if (y < height && x < width && abs(v - 0.5 * sinf(u * freq + time)) <= 0.001f) {
        //pixel = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
        buffer[y * width + x] = float4{ 1.0f, 0.0f, 0.0f, 0.0f };
    }
    else {
        //buffer[y * width + x] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    //surf2Dwrite(pixel, tex, x * 16, y);
    //framebuffer_to_tex(tex, width, height, x, y, buffer);
}

void AllocateFrameBuffer(size_t width, size_t height) {
    frameBuffer[0].allocate_memory(width * height * sizeof(float4));
    frameBuffer[1].allocate_memory(width * height * sizeof(float4));
    frameBuffer[2].allocate_memory(width * height * sizeof(float4));
}

void ReleaseFrameBuffer() {
    frameBuffer[0].free_memory();
    frameBuffer[1].free_memory();
    frameBuffer[2].free_memory();
}

// The host CPU Sinewave thread spawner
void CUDAWriteToTex(size_t width, size_t height,
    cudaSurfaceObject_t cudaDevVertptr, cudaStream_t streamToRun,
                       float AnimTime, size_t currentFrame) {
    dim3 block(16, 16, 1);
    dim3 grid(UPPER_ALIGN(width, 16) / 16, UPPER_ALIGN(height, 16) / 16, 1);
    cudaMemsetAsync(frameBuffer[currentFrame].data(), 0, width * height * sizeof(float4), streamToRun);
    cudaStreamSynchronize(streamToRun);
    write_to_tex_kernel <<<grid, block, 0, streamToRun>>>(width,
        height, AnimTime, frameBuffer[currentFrame].data());
    cudaStreamSynchronize(streamToRun);
    framebuffer_to_tex << <grid, block, 0, streamToRun >> > (cudaDevVertptr, width,
        height, frameBuffer[currentFrame].data());
    getLastCudaError("circle_genTex_kernel execution failed.\n");
}
