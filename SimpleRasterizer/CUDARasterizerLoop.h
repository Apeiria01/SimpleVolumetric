

#pragma once

#include "DX12WindowLoop.h"
//#include "ShaderStructs.h"
#include "Device.h"
#include "CUDAFrameBuffer.h"
#include "SimplePixelShader.h"
#include "CUPipeline.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;
using Device::FrameCount;
class CUDARasterizerLoop : public DX12WindowLoop {
public:
	CUDARasterizerLoop(UINT width, UINT height, std::string name);

	virtual void OnInit();
	virtual void OnRender();
	virtual void OnDestroy();

private:

	//static const UINT FrameCount = 3;
	// Pipeline objects.
	D3D12_VIEWPORT m_viewport;
	CD3DX12_RECT m_scissorRect;
	ComPtr<IDXGISwapChain3> m_swapChain;
	//ComPtr<ID3D12Device4> m_device;
	ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
	ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];
	ComPtr<ID3D12CommandQueue> m_commandQueue;
	ComPtr<ID3D12RootSignature> m_rootSignature;
	ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
	ComPtr<ID3D12DescriptorHeap> m_samplerHeap;
	ComPtr<ID3D12DescriptorHeap> m_textureHeap;
	ComPtr<ID3D12PipelineState> m_pipelineState;
	ComPtr<ID3D12GraphicsCommandList> m_commandList;

	UINT m_rtvDescriptorSize;

	// App resources.
	ComPtr<ID3D12Resource> m_texturedVertex;
	//ComPtr<ID3D12Resource> m_textureBuffer;
	D3D12_VERTEX_BUFFER_VIEW m_texturedBufferView;
	CUDAFrameBuffer* frameBuffer;

	// Synchronization objects.
	UINT m_frameIndex;
	HANDLE m_fenceEvent;
	ComPtr<ID3D12Fence> m_fence;
	UINT64 m_fenceValues[FrameCount];

	GPUMemory<ColoredVertexData> m_CUDAVertex;
	CUPipeline m_CUDAPipeline;

	// CUDA objects
	cudaExternalMemoryHandleType m_externalMemoryHandleType;
	//cudaExternalMemory_t m_externalMemory;
	cudaExternalSemaphore_t m_externalSemaphore;
	//cudaStream_t m_streamToRun[FrameCount];

	//cudaSurfaceObject_t m_cuSurface;
	//LUID m_dx12deviceluid;
	//UINT m_cudaDeviceID;
	UINT m_nodeMask;
	float m_AnimTime;
	void* m_cudaDevVertptr = NULL;

	void LoadPipeline();
	//void InitCuda();
	void LoadAssets();
	void PopulateCommandList();
	void MoveToNextFrame();
	void WaitForGpu();
};
