#include "CUPipeline.h"
#include "device_launch_parameters.h"

#define UPPER_ALIGN(A,B) ((UINT)(((A)+((B)-1))&~(B - 1)))
CUPipeline::~CUPipeline() {
	m_LocalTriangleArray.free_memory();
}

__device__ float Min3(float a, float b, float c) {
	float min = a;
	if (b < min) min = b;
	if (c < min) min = c;
	return min;
}

__device__ float Max3(float a, float b, float c) {
	float max = a;
	if (b > max) max = b;
	if (c > max) max = c;
	return max;
}

__device__ void ScanlineRasterize(NaniteStyleTrianglePrimitive& triangle, Array4f* frameBuffer, UINT width, UINT height) {
	int y0 = triangle.boundingBoxMin.y();
	int y1 = triangle.boundingBoxMax.y();
	
	Array3f edgeABC = { triangle.edge_A2B.y(), triangle.edge_B2C.y(), triangle.edge_C2A.y() };
	Array3f invEdgeABC = { 
		triangle.edge_A2B.y() == 0 ? 1e8 : 1 / triangle.edge_A2B.y(), 
		triangle.edge_B2C.y() == 0 ? 1e8 : 1 / triangle.edge_B2C.y(),
		triangle.edge_C2A.y() == 0 ? 1e8 : 1 / triangle.edge_C2A.y()
	};
	float CY0 = triangle.C0, CY1 = triangle.C1, CY2 = triangle.C2;
	Array3f crossX;
	Array3f MinX;
	Array3f MaxX;
	float xMin = triangle.boundingBoxMin.x(), xMax = triangle.boundingBoxMax.x();
	while (true) {
		crossX = { CY0, CY1, CY2 }; 
		crossX *= invEdgeABC;
		MinX = {
			edgeABC.x() < 0 ? crossX.x() : 0.0f,
			edgeABC.y() < 0 ? crossX.y() : 0.0f,
			edgeABC.z() < 0 ? crossX.z() : 0.0f
		};
		MaxX = {
			edgeABC.x() < 0 ? xMax - xMin : crossX.x(),
			edgeABC.y() < 0 ? xMax - xMin : crossX.y(),
			edgeABC.z() < 0 ? xMax - xMin : crossX.z()
		};
		float x0 = ceil(Max3(MinX.x(), MinX.y(), MinX.z()));
		float x1 = (Min3(MaxX.x(), MaxX.y(), MaxX.z()));
		x0 += xMin;
		x1 += xMin;
		
		for (float x = x0; x <= x1; x++) {
			if (x < width && y0 < height) {
				frameBuffer[(int)y0 * width + (int)x] = { 1.0f, 0.0f, 0.0f, 0.0f };
			}
		}
		if (y0 >= y1) break;
		y0++;
		CY0 += triangle.edge_A2B.x();
		CY1 += triangle.edge_B2C.x();
		CY2 += triangle.edge_C2A.x();
	}
}

__global__ void primitiveAssemblyKernel(ColoredVertexData* vertex, VertexTrianglePrimitive* out, UINT numVertex) {
	auto vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexIdx < numVertex) {
		out[vertexIdx / 3].vertexData[vertexIdx % 3] = {vertex[vertexIdx]};
	}
}

__device__ __forceinline__ float __fClamp(float f, float min = 0.0f, float max = 1.0f) {
	float rtn = f;
	rtn = f > max ? max : f;
	rtn = f < min ? min : f;
	return rtn;
}

__global__ void rasterizeKernel(VertexTrianglePrimitive* in, Array4f* frameBuffer, UINT numTriangle, UINT width, UINT height) {
	auto triangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (triangleIdx < numTriangle) {
		VertexTrianglePrimitive triangle = in[triangleIdx];
		NaniteStyleTrianglePrimitive tri ={};
		Vector2f A = { triangle[0].position.x(), triangle[0].position.y() };
		Vector2f B = { triangle[1].position.x(), triangle[1].position.y() };
		Vector2f C = { triangle[2].position.x(), triangle[2].position.y() };

		tri.edge_A2B = B - A;
		tri.edge_B2C = C - B;
		tri.edge_C2A = A - C;

		Vector4f e_AB_4 = triangle[1].position - triangle[0].position;
		Vector4f e_AC_4 = triangle[2].position - triangle[0].position;

		volatile float twice_tri_area = e_AB_4.x() * e_AC_4.y() - e_AB_4.y() * e_AC_4.x();
		volatile bool valid = twice_tri_area > 0;
		volatile float rcp_twice_tri_area = twice_tri_area == 0 ? 1e8f : 1 / twice_tri_area;

		float depth_grad_x = (e_AB_4.y() * e_AC_4.z() - e_AB_4.z() * e_AC_4.y()) * rcp_twice_tri_area;
		float depth_grad_y = (e_AB_4.x() * e_AC_4.z() - e_AB_4.z() * e_AC_4.x()) * rcp_twice_tri_area;

		int xMin = (int)Min3(triangle[0].position.x(), triangle[1].position.x(), triangle[2].position.x());
		int xMax = (int)Max3(triangle[0].position.x(), triangle[1].position.x(), triangle[2].position.x());
		int yMin = (int)Min3(triangle[0].position.y(), triangle[1].position.y(), triangle[2].position.y());
		int yMax = (int)Max3(triangle[0].position.y(), triangle[1].position.y(), triangle[2].position.y());

		tri.boundingBoxMax = { xMax, yMax };
		tri.boundingBoxMin = { xMin, yMin };

		Vector2f minCor = { xMin, yMin };

		Vector2f v0 = Vector2f(triangle[0].position.x(), triangle[0].position.y()) - minCor;
		Vector2f v1 = Vector2f(triangle[1].position.x(), triangle[1].position.y()) - minCor;
		Vector2f v2 = Vector2f(triangle[2].position.x(), triangle[2].position.y()) - minCor;
		
		tri.C0 = v0.x() * tri.edge_A2B.y() - v0.y() * tri.edge_A2B.x();
		tri.C1 = v1.x() * tri.edge_B2C.y() - v1.y() * tri.edge_B2C.x();
		tri.C2 = v2.x() * tri.edge_C2A.y() - v2.y() * tri.edge_C2A.x();

		tri.C0 -= __fClamp(tri.edge_A2B.y() + __fClamp(1.0f - tri.edge_A2B.x()));
		tri.C1 -= __fClamp(tri.edge_B2C.y() + __fClamp(1.0f - tri.edge_B2C.x()));
		tri.C2 -= __fClamp(tri.edge_C2A.y() + __fClamp(1.0f - tri.edge_C2A.x()));

		float depth_0_0 = e_AB_4.z() - e_AB_4.x() * depth_grad_x - e_AB_4.y() * depth_grad_y; // the depth value of left up corner
		tri.depth_plane = {depth_grad_x, depth_grad_y, depth_0_0};

		ScanlineRasterize(tri, frameBuffer, width, height);
	}
}

CUPipeline::CUPipeline() {
}

void CUPipeline::setFrameBufferAndStream(GPUMemory<Array4f>* gpuFrameBuffer, cudaStream_t stream) {
	m_ExternalFrameBuffer = gpuFrameBuffer;
	m_stream = stream;
}

void CUPipeline::setRenderTargetSize(UINT renderTargetWidth, UINT renderTargetHeight)
{
	m_width = renderTargetWidth;
	m_height = renderTargetHeight;
}

void CUPipeline::setPipelineResource(GPUMemory<ColoredVertexData>* ExternalVertexArray, GPUMemory<UINT>* ExternalIndexArray) {
	m_ExternalVertexArray = ExternalVertexArray;
	m_ExternalIndexArray = ExternalIndexArray;
	if (m_LocalTriangleArray.size() < m_ExternalVertexArray->size()) {
		m_LocalTriangleArray.resize(m_ExternalVertexArray->size() * 4);
	}
}

void CUPipeline::primitiveAssembly(UINT numVertex) {
	UINT numTriangle = numVertex / 3;
	dim3 block(256, 1, 1);
	dim3 grid(UPPER_ALIGN(numVertex, 256) / 256, 1, 1);
	primitiveAssemblyKernel << <grid, block, 0, m_stream >> > (m_ExternalVertexArray->data(), m_LocalTriangleArray.data(), numVertex);
}

void CUPipeline::rasterize(UINT num) {
	UINT numVertex = num;
	UINT numTriangle = numVertex / 3;
	dim3 block(256, 1, 1);
	dim3 grid(UPPER_ALIGN(numTriangle, 256) / 256, 1, 1);
	rasterizeKernel << <grid, block, 0, m_stream >> > (m_LocalTriangleArray.data(), m_ExternalFrameBuffer->data(), numTriangle, m_width, m_height);
}

