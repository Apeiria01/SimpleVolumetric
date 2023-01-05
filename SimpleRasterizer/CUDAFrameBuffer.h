#pragma once
#include <Eigen/Core>
#include "GPUMemory.h"
#include "DXTexture.h"
using Eigen::Array4f;
using Eigen::Array2i;

#include "Device.h"
using Device::FrameCount;

class CUDAFrameBuffer {
public:
	Array4f* getRaw(int i);
	inline GPUMemory<Array4f>* getWrap(int i) {
		return &frameBuffer[i % FrameCount];
	}
	CUDAFrameBuffer(size_t width, size_t height, D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle, D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle);
	~CUDAFrameBuffer();
	void WriteToTex(cudaStream_t stream, UINT frameIndex);
	void Clear(cudaStream_t stream, UINT frameIndex);
	inline cudaSurfaceObject_t getSurface() {
		return m_texture.getCudaTex();
	}
	inline D3D12_CPU_DESCRIPTOR_HANDLE getCPUPointer() {
		return m_texture.getCPUDescriptor();
	}
	inline D3D12_GPU_DESCRIPTOR_HANDLE getGPUPointer() {
		return m_texture.getGPUDescriptor();
	}
	inline Array2i getResolution() {
		return res;
	}
private:

	GPUMemory<Array4f> frameBuffer[FrameCount];
	DXTexture m_texture;
	Array2i res;
};
