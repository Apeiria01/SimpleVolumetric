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

struct Material {
	Array3f albedo;
	float roughness;
	float metalness;
	Array3f emission;
};
