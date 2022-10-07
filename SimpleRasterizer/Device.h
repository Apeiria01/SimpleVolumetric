#pragma once
#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_5.h>
#include <driver_types.h>
using namespace Microsoft::WRL;
namespace Device {
	constexpr UINT FrameCount = 3u;
	extern ComPtr<ID3D12Device4> DX12Device;
	extern ComPtr<IDXGIFactory5> DXfactory;
	extern cudaStream_t Streams[FrameCount];
	extern UINT NodeMask;
	
	extern void InitDevice(HWND hWnd);
}