#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "TracerFunc.h"
constexpr UINT SphereNum = 7u;
constexpr float m_PI = 3.1415926535f;

__constant__ float deviceViewMat[sizeof(Eigen::Matrix4f) / sizeof(float)];

__constant__ float SphereArr[sizeof(Sphere) * SphereNum / sizeof(float)];

__constant__ float MaterialArr[sizeof(Material) * SphereNum / sizeof(float)];

__device__ __forceinline float CosinePDF(float dot) {
	return dot / m_PI;
}

void CopyMatForR(const Eigen::Matrix4f* mat, cudaStream_t streamToRun) {
    checkCudaErrors(
        cudaMemcpyToSymbolAsync(deviceViewMat, mat, sizeof(Eigen::Matrix4f), 0Ui64, cudaMemcpyHostToDevice, streamToRun)
    );
    return;
}

__device__ __forceinline float GGX(float NdotV, float roughness)
{
    float a4 = (roughness * roughness) * (roughness * roughness);
    float denom = NdotV * NdotV * (a4 - 1.0f) + 1.0f;
    return a4 / (m_PI * denom * denom);
}

__device__ __forceinline float SchlickGGX(float NoL, float NoV, float roughness)
{
    float k = roughness + 1.0f;
    k *= k * 0.125f;
    float gl = NoL / (NoL * (1.0f - k) + k);
    float gv = NoV / (NoV * (1.0f - k) + k);
    return gl * gv;
}

__device__ __forceinline Eigen::Vector4f CookTorranceBRDF(float NoL, float NoV, float NoH, float VoH, const Eigen::Vector4f& F, float roughness)
{
    Eigen::Vector4f DFG = F * GGX(NoH, roughness) * SchlickGGX(NoL, NoV, roughness);
    float denom = 4.0 * NoL * NoV + 0.0001;
    return DFG / denom;
}

__device__ int ClosestHitCheck(const CastedRay& ray, Eigen::Vector4f* position, Eigen::Vector4f* normal, Material* hitMaterial, float* hitCoef) {
    auto sceneDef = (Sphere*)(&SphereArr);
    auto materialDef = (Material*)(&MaterialArr);
    float coef = 999999999.0f;
    int minIdx = -1;
    for (int i = 0; i < SphereNum; i++) {
        Eigen::Vector4f sphere = sceneDef[i].positionAndRadius;
        float radius = sphere[3];
        sphere[3] = 0.0f;
        float XDcosTheta = (sphere - ray.position).dot(ray.direction); //height
        float X2D2sinTheta2 = (sphere - ray.position).dot(sphere - ray.position) - XDcosTheta * XDcosTheta;
        if (X2D2sinTheta2 > radius * radius) {
            continue;
        }
        else
        {
            float deltaCoef = sqrt(radius * radius - X2D2sinTheta2);

            float coef1 = XDcosTheta - deltaCoef;
            float coef2 = XDcosTheta + deltaCoef;
            float nearest = min(coef1, coef2);
            if (nearest > 0.0f && nearest < coef) {
                coef = nearest;
                minIdx = i;
            } 
        }
    }
    if (minIdx >= 0) {
        Eigen::Vector4f sphere = sceneDef[minIdx].positionAndRadius;
        float radius = sphere[3];
        sphere[3] = 0.0f;
        *position = ray.position + coef * ray.direction;
        *normal = (*position - sphere).normalized();
        *hitCoef = coef;
        *hitMaterial = materialDef[minIdx];
    }
    return minIdx >= 0;
}

__device__ __forceinline Eigen::Matrix4f GetBasis(const Eigen::Vector4f& n)
{
    // Make vector q that is non-parallel to n
    Eigen::Vector4f q = n;
    Eigen::Vector4f aq = {abs(q[0]), abs(q[1]), abs(q[2]), abs(q[3]) };
    if (aq.x() <= aq.y() && aq.x() <= aq.z()) {
        q[0] = 1.0f;
    }
    else if (aq.y() <= aq.x() && aq.y() <= aq.z()) {
        q[1] = 1.0f;
    }
    else {
        q[2] = 1.0f;
    }
    // Generate two vectors perpendicular to n
    Eigen::Vector4f t = q.cross3(n).normalized();
    Eigen::Vector4f b = n.cross3(t).normalized();

    // Construct the rotation matrix
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m << t, b, n, 
        Eigen::Vector4f{ 0.0f, 0.0f, 0.0f, 1.0f };//column
    return m;
}

__device__ __forceinline Eigen::Vector4f SampleHemisphere(curandState_t* randState) {
    Eigen::Array2f arr = { (curand_uniform(randState) * 2.0f - 1.0f), (curand_uniform(randState) * 2.0f - 1.0f) };
    float r = sqrt(arr[0]);
    float theta = 2.0f * m_PI * arr[1];
    return Eigen::Vector4f{ r * cos(theta), r * sin(theta), sqrt(__saturatef(1.0f - arr[0])), 0.0f }.normalized();
}

__global__ void TracerPixelShaderDevice(unsigned int width,
    unsigned int height, Array4f* buffer
    ) {
    // composite ray
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr UINT maxSamplePerPixel = 128u;
    constexpr UINT maxBounce = 5u;
    constexpr float pCont = 0.75f;

    const Eigen::Vector4f& boxMin = { -1.0f, -1.0f, -1.0f, 1.0f };
    const Eigen::Vector4f& boxMax = { 1.0f, 1.0f, 1.0f, 1.0f };
    CastedRay r;
    auto mat = (Eigen::Matrix4f*)(&deviceViewMat);
    
    //Store scene by hard coding
    
    if (y < height && x < width) {
        
        curandState_t randState;
        for (UINT i = 0; i < maxSamplePerPixel; i++) {
            curand_init(y * width + x, 0, 0, &randState);
            float randomShiftX = (curand_uniform(&randState) * 2.0f - 1.0f) / 2.0f;
            float randomShiftY = (curand_uniform(&randState) * 2.0f - 1.0f) / 2.0f;

            float u = ((x + randomShiftX + 0.5f) / (float)width) * 2.0f - 1.0f;
            float v = ((y + randomShiftY + 0.5f) / (float)height) * 2.0f - 1.0f;

            r.position = (*mat) * Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
            r.direction = (*mat) * Eigen::Vector4f(u / height * width, v, 1.0f, 0.0f);
            r.direction = r.direction.normalized(); //Necessary process

            for (UINT j = 0; j < maxBounce; j++) {
                Eigen::Vector4f position = {};
                Eigen::Vector4f normal = {};
                Material hitMaterial = {};
                float hitCoef = 0.0f;
                int hitRes = ClosestHitCheck(r, &position, &normal, &hitMaterial, &hitCoef);
                if (!hitRes) {// No hit, fill with blank color
                    break;
                }
                else
                {

                }

                float terminateRoll = curand_uniform(&randState);
                if (terminateRoll) break;
            }
            
        }

        
    }
    else
    {

    }

}