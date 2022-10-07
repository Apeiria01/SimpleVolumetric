#pragma once
#include "GPUMemory.h"
template<typename dataType>
class CUDAVertexDataBuffer {
public:

private:
	GPUMemory<dataType> m_inVertexData;
	GPUMemory<dataType> m_outTriangleData;
	
};