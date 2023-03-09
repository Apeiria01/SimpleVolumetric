#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "SimplePixelShader.h"
#include "GPUMemory.h"

//__constant__ Eigen::Matrix4f deviceViewMat;
__constant__ float deviceViewMat[sizeof(Eigen::Matrix4f) / sizeof(float)];

__global__ void SimplePixelShaderDevice(unsigned int width,
    unsigned int height, float time, Array4f* buffer) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    float freq = 4.0f;
    if (y < height && x < width && abs(v - 0.5 * sinf(u * freq + time)) <= 0.001f) {
        buffer[y * width + x] <<  1.0f, 0.0f, 0.0f, 0.0f;
    }
    else {
        //buffer[y * width + x] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
}

__device__ __forceinline Eigen::Vector4f __fminf(const Eigen::Vector4f& a, const Eigen::Vector4f& b) {
    return Eigen::Vector4f(min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]));
}

__device__ __forceinline Eigen::Vector4f __fmaxf(const Eigen::Vector4f& a, const Eigen::Vector4f& b) {
    return Eigen::Vector4f(max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]));
}

__device__ __forceinline Eigen::Array4f transfer(float sigma) {//RGBA
    return Eigen::Array4f{1.0f, 0.0f, 0.0f, 0.0f} * sigma + Eigen::Array4f{ 0.0f, 0.0f, 1.0f, 0.0f } * (1.0f - sigma);
}

__device__ int IntersectAABB(const CastedRay& r, const Eigen::Vector4f& boxMin, const Eigen::Vector4f& boxMax, float* tnear, float* tfar) {
    Eigen::Array4f invR = Eigen::Array4f(1.0f) / Eigen::Array4f(r.direction);
    Eigen::Vector4f tbot = invR * Eigen::Array4f(boxMin - r.position);
    Eigen::Vector4f ttop = invR * Eigen::Array4f(boxMax - r.position);
    
    // re-order intersections to find smallest and largest on each axis
    auto tmin = __fminf(ttop, tbot);
    auto tmax = __fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin[0], tmin[1]), fmaxf(tmin[0], tmin[2]));
    float smallest_tmax = fminf(fminf(tmax[0], tmax[1]), fminf(tmax[0], tmax[2]));
    *tnear = largest_tmin;
    *tfar = smallest_tmax;
    return smallest_tmax > largest_tmin;
}

__device__ float dydTarget(cudaTextureObject_t volumetricData, Eigen::Vector4f base, float step) {

}

__global__ void VolumetricPixelShaderDevice(unsigned int width,
    unsigned int height, Array4f* buffer, 
    cudaTextureObject_t volumetricData, cudaTextureObject_t transferFunctionTex) {
    // composite ray
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr float gradStep = 0.005f;
    constexpr UINT maxIterationStep = 128u;
    float u = (x / (float)width) * 2.0f - 1.0f;
    float v = (y / (float)height) * 2.0f - 1.0f;
    const Eigen::Vector4f& boxMin = { -1.0f, -1.0f, -1.0f, 1.0f };
    const Eigen::Vector4f& boxMax = { 1.0f, 1.0f, 1.0f, 1.0f };
    CastedRay r;
    auto mat = (Eigen::Matrix4f*)(&deviceViewMat);
    r.position = (*mat) * Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
    r.direction = (*mat) * Eigen::Vector4f(u / height * width, v, 1.0f, 0.0f);
    r.direction = r.direction.normalized();
    float tNear, tFar;
    int hit = IntersectAABB(r, boxMin, boxMax, &tNear, &tFar);
    Eigen::Array4f colorSum = {0.0f, 0.0f, 0.0f, 1.0f};
    if (y < height && x < width && hit && tFar > 0.0f) {
        if (tNear < 0.0f) tNear = 0.0f;
        float step = (tFar - tNear) / maxIterationStep;
        UINT i = 0;
        //float tBase = 1.0f;
        for (i = 0; i < maxIterationStep; i++) 
        {
            float t = ((i + 0.5f) * step + tNear);
            Eigen::Vector4f samplePos = r.position + r.direction * t;
            Eigen::Array3f texturePosUVW = {
                (samplePos[0] - boxMin[0]) / (boxMax[0] - boxMin[0]),
                (samplePos[1] - boxMin[1]) / (boxMax[1] - boxMin[1]),
                (samplePos[2] - boxMin[2]) / (boxMax[2] - boxMin[2])
            };
            float sample = tex3D<float>(volumetricData, texturePosUVW[0], texturePosUVW[1], texturePosUVW[2]);
            float4 c0 = tex1D<float4>(transferFunctionTex, (sample));
            float trans = __expf(-step / r.direction.norm() * sample * 16.0f);
            Eigen::Array4f col = { c0.x, c0.y, c0.z, 0.0f };
            //Eigen::Array4f col = transfer(sample);
            colorSum = colorSum + col * colorSum[3] * (1.0f - trans);
            colorSum[3] *= trans;
        }
        buffer[y * width + x] = colorSum;
    }
    else
    {
        
    }
    
}

void CopyMat(const Eigen::Matrix4f* mat, cudaStream_t streamToRun) {
    checkCudaErrors(
        cudaMemcpyToSymbolAsync(deviceViewMat, mat, sizeof(Eigen::Matrix4f), 0Ui64, cudaMemcpyHostToDevice, streamToRun)
    );
    return;
}

void VolumetricPixelShader(size_t width, size_t height,
    cudaStream_t streamToRun,
    Array4f* buffer, cudaTextureObject_t volumetricData, cudaTextureObject_t transferFunctionTex) {
    dim3 block(16, 16, 1);
    dim3 grid(UPPER_ALIGN(width, 16) / 16, UPPER_ALIGN(height, 16) / 16, 1);
    VolumetricPixelShaderDevice << <grid, block, 0, streamToRun >> > (width, height, buffer, volumetricData, transferFunctionTex);
    getLastCudaError("circle_genTex_kernel execution failed.\n");
}

void SimplePixelShader(size_t width, size_t height,
    cudaStream_t streamToRun,
    float AnimTime, size_t currentFrame, Array4f* buffer) {
    dim3 block(16, 16, 1);
    dim3 grid(UPPER_ALIGN(width, 16) / 16, UPPER_ALIGN(height, 16) / 16, 1);
    SimplePixelShaderDevice << <grid, block, 0, streamToRun >> > (width, height, AnimTime, buffer);
    getLastCudaError("circle_genTex_kernel execution failed.\n");
}