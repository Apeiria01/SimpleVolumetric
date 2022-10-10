#pragma once
#include "GPUMemory.h"
#include <Eigen/Core>
#include <wrl.h>
#include "CUDAFrameBuffer.h"
using Eigen::Array4f;
struct ColoredVertexData {
	Array4f position;
	Array4f color;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct ColoredPixelData {
	Array4f positionCameraSpace;
	Array4f color;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct ColoredVertexTrianglePrimitive {
	ColoredVertexData vertexData[3];
};

class CUPipeline {
public:
	CUPipeline(CUDAFrameBuffer* buffer);
	~CUPipeline();
	void setFrameBufferAndStream();
	void setPipelineResource(GPUMemory<ColoredVertexData>* ExternalVertexArray, GPUMemory<UINT>* ExternalIndexArray);
	void primitiveAssembly(cudaStream_t stream, UINT numVertex);
	void rasterize(cudaStream_t stream, UINT num, UINT frameNum, UINT width, UINT height);
private:
	GPUMemory<ColoredVertexData>* m_ExternalVertexArray;
	GPUMemory<UINT>* m_ExternalIndexArray;
	GPUMemory<ColoredVertexTrianglePrimitive> m_LocalTriangleArray;
	CUDAFrameBuffer* m_frameBuffer;
	UINT currentStage;
};