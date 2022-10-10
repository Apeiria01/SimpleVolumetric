#include "CUPipeline.h"
#include "device_launch_parameters.h"
#define UPPER_ALIGN(A,B) ((UINT)(((A)+((B)-1))&~(B - 1)))
CUPipeline::~CUPipeline() {
	m_LocalTriangleArray.free_memory();
}

__global__ void primitiveAssemblyKernel(ColoredVertexData* vertex, ColoredVertexTrianglePrimitive* out, UINT numVertex) {
	auto vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexIdx < numVertex) {
		
		out[vertexIdx / 3].vertexData[vertexIdx % 3] = {vertex[vertexIdx]};
	}
}

__global__ void rasterizeKernel(ColoredVertexTrianglePrimitive* in, Array4f* frameBuffer, UINT numTriangle, UINT width, UINT height) {
	auto triangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (triangleIdx < numTriangle) {
		ColoredVertexTrianglePrimitive triangle = in[triangleIdx];
		int minY = height, maxY = 0;
		int minX = width, maxX = 0;
#pragma unroll
		for (int i = 0; i < 3; i++) {
			auto xHere = triangle.vertexData[i].position.x();
			auto yHere = triangle.vertexData[i].position.y();
			if (minX > xHere) minX = xHere;
			if (maxX < xHere) maxX = xHere;
			if (minY > yHere) minY = yHere;
			if (maxY < yHere) maxY = yHere;
		}

		for (int i = minX; i <= maxX; i++) {
			for (int j = minY; j <= maxY + 64; j++) {

				frameBuffer[j * width + i] = { 1.0f * (i - minX) / 256.0f * (j - minY) / 256.0f,0.0f,0.0f,0.0f };
			}
		}
	}
}

CUPipeline::CUPipeline(CUDAFrameBuffer* buffer) {
	m_frameBuffer = buffer;
}

void CUPipeline::setPipelineResource(GPUMemory<ColoredVertexData>* ExternalVertexArray, GPUMemory<UINT>* ExternalIndexArray) {
	m_ExternalVertexArray = ExternalVertexArray;
	m_ExternalIndexArray = ExternalIndexArray;
	if (m_LocalTriangleArray.size() < m_ExternalVertexArray->size()) {
		m_LocalTriangleArray.resize(m_ExternalVertexArray->size() * 4);
	}
}

void CUPipeline::primitiveAssembly(cudaStream_t stream, UINT numVertex) {
	UINT numVertex = numVertex;
	UINT numTriangle = numVertex / 3;
	dim3 block(256, 1, 1);
	dim3 grid(UPPER_ALIGN(numVertex, 256) / 256, 1, 1);
	primitiveAssemblyKernel << <grid, block, 0, stream >> > (m_ExternalVertexArray->data(), m_LocalTriangleArray.data(), numVertex);
}

void CUPipeline::rasterize(cudaStream_t stream, UINT num, UINT frameNum, UINT width, UINT height) {
	UINT numVertex = num;
	UINT numTriangle = numVertex / 3;
	dim3 block(256, 1, 1);
	dim3 grid(UPPER_ALIGN(numTriangle, 256) / 256, 1, 1);
	rasterizeKernel << <grid, block, 0, stream >> > (m_LocalTriangleArray.data(), m_frameBuffer->getRaw(frameNum), numTriangle, width, height);
}

