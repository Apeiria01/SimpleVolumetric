#pragma once


#include <dxgi1_4.h>

#include <DirectXMath.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <Eigen/Core>

#define UPPER_ALIGN(A,B) ((UINT)(((A)+((B)-1))&~(B - 1)))

using namespace DirectX;
using Eigen::Array4f;
using Eigen::Array3f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using Eigen::Array2f;
using Eigen::Array2i;
using Eigen::Vector2i;
using Eigen::Vector2f;
using Eigen::Matrix4f;
struct ColoredVertexData {
	Vector4f position;
	Array4f color;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct CastedRay {
	Vector4f position;
	Vector4f direction;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

#define stol1(x) (x <= 0.04045f ? x / 12.92f : pow((x + 0.055f) / 1.055f, 2.4f))
#define stol3(x, y, z) Eigen::Array3f{stol1(x), stol1(y), stol1(z)}
#define ltos1(x) (x <= 0.0031308f ? x * 12.92f : 1.055f * pow(x, 0.4166667f) - 0.055f)
#define ltos3(x, y, z) Eigen::Array3f{ltos1(x), ltos1(y), ltos1(z)}


struct Sphere {
	Vector4f positionAndRadius;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	__host__ __device__ Sphere(float x, float y, float z, float r) {
		positionAndRadius = { x,y,z,r };
	}
};

struct ColoredPixelData {
	Array4f positionCameraSpace;
	Array4f color;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct DXTexturedVertex
{
    XMFLOAT4 position;
    XMFLOAT2 texCoord;
};

constexpr UINT SphereNum = 7u;

struct Material {
	Array3f albedo;
	float roughness;
	float metalness;
	__host__ __device__ Material(Array3f a, float r, float m) {
		albedo = a;
		roughness = r;
		metalness = m;
	}

	__host__ __device__ Material() {
		albedo = Array3f(0.0f);
		roughness = 0.0f;
		metalness = 0.0f;
	}
};
