#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "TracerFunc.h"
#include <Eigen/Geometry>

__device__ constexpr float m_PI = 3.1415926535f;
__device__ constexpr float m_extinction = 0.001256f;
__device__ constexpr float m_scatter = 0.7812f;

__constant__ float deviceViewMat[sizeof(Eigen::Matrix4f) / sizeof(float)];

__constant__ float SphereArr[sizeof(Sphere) * SphereNum / sizeof(float)];

__constant__ float MaterialArr[sizeof(Material) * SphereNum / sizeof(float)];

__device__ __forceinline float CosinePDF(float dot) {
	return dot / m_PI;
}

__device__ __forceinline float PhaseFunction(float theta) {
    return 1 / (4 * m_PI);
}

__device__ __forceinline float MediaPDF(float distance) {
    return m_extinction * exp(-distance * m_extinction);
}

__device__ __forceinline Eigen::Array3f Fresnel(float VoH, const Eigen::Array3f& F0)
{
    return F0 + (1.0 - F0) * (1.0f - VoH) * (1.0f - VoH) * (1.0f - VoH) * (1.0f - VoH) * (1.0f - VoH);
}

__device__ __forceinline float TRGGX(float NdotH, float roughness)
{
    float a2 = (roughness * roughness);
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (m_PI * denom * denom);
}

__device__ __forceinline float SGGX(float NoL, float NoV, float roughness)
{
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    float gl = NoL / (NoL * (1.0f - k) + k);
    float gv = NoV / (NoV * (1.0f - k) + k);
    return gl * gv;
}

__device__ __forceinline Eigen::Array3f CookTorranceBRDF(float NoL, float NoV, float NoH, float VoH, const Eigen::Array3f& F, float roughness)
{
    Eigen::Array3f DFG = F * TRGGX(NoH, roughness) * SGGX(NoL, NoV, roughness);
    return DFG / (4.0f * NoL * NoV + 0.0001f);
}

__device__ Eigen::Array3f CalcBRDF(const Eigen::Vector4f& n, const Eigen::Vector4f& v, const Eigen::Vector4f& l, const Material& m)
{
    float NoV = n.dot(v);
    float NoL = n.dot(l);
    Eigen::Vector4f h = (v + l).normalized();
    float NoH = n.dot(h);
    float VoH = v.dot(h);

    Eigen::Array3f F0 = Eigen::Array3f(0.04f) * (1.0f - m.metalness) + m.metalness * m.albedo;
    Eigen::Array3f F = Fresnel(VoH, F0);
    Eigen::Array3f Kd = (1.0f - F) * (1.0f - m.metalness);

    return (Kd * (m.albedo / m_PI) + CookTorranceBRDF(NoL, NoV, NoH, VoH, F, m.roughness)) * NoL;
}

void CopyMatForR(const Eigen::Matrix4f* mat, cudaStream_t streamToRun) {
    checkCudaErrors(
        cudaMemcpyToSymbolAsync(deviceViewMat, mat, sizeof(Eigen::Matrix4f), 0Ui64, cudaMemcpyHostToDevice, streamToRun)
    );
    return;
}


void CopySceneData(const Sphere* sph, UINT size, cudaStream_t streamToRun) {
    checkCudaErrors(
        cudaMemcpyToSymbolAsync(SphereArr, &sph[0], sizeof(Sphere) * SphereNum, 0Ui64, cudaMemcpyHostToDevice, streamToRun)
    );
    return;
}

void CopyMaterialData(const Material* m, UINT size, cudaStream_t streamToRun) {
    checkCudaErrors(
        cudaMemcpyToSymbolAsync(MaterialArr, &m[0], sizeof(Material) * SphereNum, 0Ui64, cudaMemcpyHostToDevice, streamToRun)
    );
    return;
}


__device__ int ClosestHitCheck(const CastedRay& ray, Eigen::Vector4f* position, Eigen::Vector4f* normal, Material* hitMaterial, float* hitCoef, float tNear, float tFar) {
    auto sceneDef = (Sphere*)(&SphereArr);
    auto materialDef = (Material*)(&MaterialArr);
    float coef = 999999999.0f;
    int minIdx = -1;
    for (int i = 0; i < SphereNum; i++) {
        Eigen::Vector4f sphere = sceneDef[i].positionAndRadius;
        float radius = sphere[3];
        sphere[3] = 1.0f;
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
            if (nearest > 0.0f && nearest < coef && nearest >= tNear && nearest <= tFar) {
                coef = nearest;
                minIdx = i;
            } 
        }
    }
    if (minIdx >= 0) {
        Eigen::Vector4f sphere = sceneDef[minIdx].positionAndRadius;
        float radius = sphere[3];
        sphere[3] = 1.0f;
        if(position) *position = ray.position + coef * ray.direction;
        if(normal) *normal = (*position - sphere).normalized();
        if(hitCoef) *hitCoef = coef;
        if(hitMaterial) *hitMaterial = materialDef[minIdx];
    }
    return minIdx >= 0;
}

__device__ Eigen::Matrix4f GetBasis(const Eigen::Vector4f& n)
{
    Eigen::Vector4f q = n;
    Eigen::Vector4f aq = q.cwiseAbs();
    if (aq.x() <= aq.y() && aq.x() <= aq.z()) {
        q[0] = 1.0f;
    }
    else if (aq.y() <= aq.x() && aq.y() <= aq.z()) {
        q[1] = 1.0f;
    }
    else {
        q[2] = 1.0f;
    }
    Eigen::Vector4f t = q.cross3(n).normalized();
    Eigen::Vector4f b = n.cross3(t).normalized();
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m << t, b, n, 
        Eigen::Vector4f{ 0.0f, 0.0f, 0.0f, 1.0f };//column
    return m;
}

__device__ Eigen::Vector4f SampleHemisphere(curandState_t* randState) {
    Eigen::Array2f arr = { (curand_uniform(randState)), (curand_uniform(randState)) };
    float r = sqrt(arr[0]);
    float theta = 2.0f * m_PI * arr[1];
    return Eigen::Vector4f{ r * cos(theta), r * sin(theta), sqrt(1.0f - arr[0]), 0.0f }.normalized();
}

__device__ Eigen::Vector4f SampleSphere(curandState_t* randState) {
    Eigen::Array2f arr = { (2.0f * curand_uniform(randState) - 1.0f), (curand_uniform(randState)) };
    float r = sqrt(abs(arr[0]));
    float theta = 2.0f * m_PI * arr[1];
    return Eigen::Vector4f{ r * cos(theta), r * sin(theta), arr[0] < 0.0f ? - sqrt(arr[0] + 1.0f) : sqrt(1.0f - arr[0]), 0.0f }.normalized();
}


__device__ Eigen::Vector4f SampleLightReciprocalPDF(curandState_t* randState, const Eigen::Vector4f& orig)
{
    
    Eigen::Vector4f res = Eigen::Vector4f((curand_uniform(randState) * 2.0f - 1.0f), 4.0f, (curand_uniform(randState) * 2.0f - 1.0f), 1.0f);
    Eigen::Vector4f dis = res - orig;
    float pdf = 1.0f * abs(Eigen::Vector4f{ 0.0f, -1.0f, 0.0f, 0.0f }.dot(dis.normalized())) / ((2.0f * 2.0f) * dis.squaredNorm());
    res[3] = pdf;
    return res;
}

__global__ void TracerPixelShaderDevice(unsigned int width,
    unsigned int height, Array4f* buffer, const UINT maxSamplePerPixel,
    size_t currentFrame) {
    // composite ray
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr UINT maxBounce = 5u;
    constexpr float pCont = 0.75f;

    const Eigen::Vector4f& boxMin = { -1.0f, -1.0f, -1.0f, 1.0f };
    const Eigen::Vector4f& boxMax = { 1.0f, 1.0f, 1.0f, 1.0f };
    CastedRay r;
    auto mat = (Eigen::Matrix4f*)(&deviceViewMat);
    
    //Store scene by hard coding
    
    if (y < height && x < width && currentFrame <= 256) {
        Eigen::Array3f cumulativeLight = Eigen::Array3f(0.0f);
        curandState_t randState;
        curand_init((y * width + x) + clock64(), 0, 0, &randState);
        for (UINT i = 0; i < maxSamplePerPixel; i++) {
            float randomShiftX = (curand_uniform(&randState) * 2.0f - 1.0f) / 1.0f;
            float randomShiftY = (curand_uniform(&randState) * 2.0f - 1.0f) / 1.0f;

            float u = ((x + randomShiftX + 0.5f) / (float)width) * 2.0f - 1.0f;
            float v = ((y + randomShiftY + 0.5f) / (float)height) * 2.0f - 1.0f;

            Eigen::Array3f powerRemain = Eigen::Array3f(1.0f);

            r.position = (*mat) * Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
            r.direction = (*mat) * Eigen::Vector4f(u / height * width, v, 1.0f, 0.0f);
            r.position = r.position + r.direction;
            r.direction = r.direction.normalized(); //Necessary process

            for (UINT j = 0; j < maxBounce; j++) {
                Eigen::Vector4f position = {};
                Eigen::Vector4f normal = {};
                Material hitMaterial = {};
                float hitCoef = 0.0f;
                int hitRes = ClosestHitCheck(r, &position, &normal, &hitMaterial, &hitCoef, 0.01f, 10000.0f);
                if (!hitRes) {// No hit, fill with blank color
                    break;
                }
                else
                {
                    float penetrateProb = exp(-m_extinction * hitCoef);
                    float penetrateRoll = curand_uniform(&randState);
                    if (penetrateRoll >= penetrateProb) {

                    }
                    else {
                        //float stopCoef = 
                    }
                    //media
                    Eigen::Vector4f neoPos = position + normal * 0.001f;
                    Eigen::Vector4f lightPos = SampleLightReciprocalPDF(&randState, neoPos);
                    float lightPdfR = lightPos[3];
                    lightPos[3] = 1.0f;
                    CastedRay ray2 = {};
                    ray2.position = neoPos;
                    ray2.direction = (lightPos - neoPos).normalized();
                    float t1 = (lightPos - neoPos).norm();
                    float t2 = 0.0f;
                    int lightCast = ClosestHitCheck(ray2, nullptr, nullptr, nullptr, &t2, 0.01f, 10000.0f);
                    if (!lightCast || t2 - t1 >= -0.01f) {
                        Array3f lightE = { 1.0f, 0.85f, 0.6f };
                        lightE *= 512.0f;
                        cumulativeLight += powerRemain * CalcBRDF(normal, -r.direction, ray2.direction, hitMaterial) * lightE * lightPdfR;
                    }
                    else
                    {
                
                    }
                    auto newDir = SampleHemisphere(&randState);
                    newDir = (GetBasis(normal) * newDir).normalized();
                    float samplePDF = CosinePDF(normal.dot(newDir));
                    powerRemain *= CalcBRDF(normal, -r.direction, newDir, hitMaterial) / (samplePDF * pCont);
                    r.direction = newDir;
                    r.position = neoPos;
                }

                float terminateRoll = curand_uniform(&randState);
                if (terminateRoll >= pCont) break;
                
            }
            
        }
        cumulativeLight = cumulativeLight / float(maxSamplePerPixel);
        cumulativeLight = ltos3(cumulativeLight[0], cumulativeLight[1], cumulativeLight[2]);
        if (currentFrame <= 3) {
            buffer[y * width + x] = { cumulativeLight[0], cumulativeLight[1] , cumulativeLight[2], 1.0f };
        }
        else if (currentFrame <= 256 * 3) {
            buffer[y * width + x] /= float(currentFrame / 3.0f) / float(currentFrame / 3.0f - 1);
            buffer[y * width + x] += Eigen::Array4f{ cumulativeLight[0], cumulativeLight[1] , cumulativeLight[2], 1.0f } / float(currentFrame / 3.0f);
        }
        else {

        }
    }
    else
    {

    }

}


void TracerPixelShader(unsigned int width,
    unsigned int height, Array4f* buffer, 
    cudaStream_t streamToRun, const UINT maxSamplePerPixel, size_t currentFrame) {
    dim3 block(16, 16, 1);
    dim3 grid(UPPER_ALIGN(width, 16) / 16, UPPER_ALIGN(height, 16) / 16, 1);
    TracerPixelShaderDevice << <grid, block, 0, streamToRun >> > (width, height, buffer, maxSamplePerPixel, currentFrame);
    getLastCudaError("circle_genTex_kernel execution failed.\n");
}