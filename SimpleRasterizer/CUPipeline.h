#pragma once
#include "GPUMemory.h"
#include <Eigen/Core>
#include <wrl.h>
#include "CUDAFrameBuffer.h"
#include "ShaderStructs.h"


struct VertexTrianglePrimitive {
	ColoredVertexData vertexData[3];
	__host__ __device__ ColoredVertexData& operator[](UINT index) {
		return vertexData[index];
	}
};

struct NaniteStyleTrianglePrimitive {
	Vector2i boundingBoxMin;
	Vector2i boundingBoxMax;
	Vector2f edge_A2B;
	Vector2f edge_B2C;
	Vector2f edge_C2A;
	float C0;
	float C1;
	float C2;
	Vector3f depth_plane;
	//Depth should be here

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


class CUPipeline {
public:
	CUPipeline();
	~CUPipeline();
	void setFrameBufferAndStream(GPUMemory<Array4f>* gpuFrameBuffer, cudaStream_t stream);
	void setPipelineResource(GPUMemory<ColoredVertexData>* ExternalVertexArray, GPUMemory<UINT>* ExternalIndexArray);
	void setRenderTargetSize(UINT renderTargetWidth, UINT renderTargetHeight);
	void primitiveAssembly(UINT numVertex);
	void rasterize(UINT num);
	void CopyMatrix(const Eigen::Matrix4f* src, cudaStream_t streamToRun);
private:
	GPUMemory<ColoredVertexData>* m_ExternalVertexArray;
	GPUMemory<UINT>* m_ExternalIndexArray;
	GPUMemory<VertexTrianglePrimitive> m_LocalTriangleArray;
	GPUMemory<Array4f>* m_ExternalFrameBuffer;
	cudaStream_t m_stream;
	UINT m_width;
	UINT m_height;
	UINT currentStage;
};