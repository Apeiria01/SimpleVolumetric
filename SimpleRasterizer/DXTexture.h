#pragma once
#include <wrl.h>
#include <d3d12.h>
#include "DXSampleHelper.h"
#include <surface_types.h>

using Microsoft::WRL::ComPtr;
class DXTexture {
public:
	DXTexture() = delete;
	DXTexture(UINT width, UINT height, D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle, D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle);
	~DXTexture();
	inline cudaSurfaceObject_t getCudaTex() {
		return m_DXTexturePTR;
	}
	inline D3D12_CPU_DESCRIPTOR_HANDLE getCPUDescriptor() {
		return m_CPUDescriptor;
	}
	inline D3D12_GPU_DESCRIPTOR_HANDLE getGPUDescriptor() {
		return m_GPUDescriptor;
	}
private:
	ComPtr<ID3D12Resource> m_textureBuffer;
	cudaSurfaceObject_t m_DXTexturePTR;
	cudaExternalMemory_t m_externalMemory;
	UINT m_width;
	UINT m_height;
	D3D12_CPU_DESCRIPTOR_HANDLE m_CPUDescriptor;
	D3D12_GPU_DESCRIPTOR_HANDLE m_GPUDescriptor;
};


