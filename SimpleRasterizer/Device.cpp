#include "Device.h"
#include <dxgi1_5.h>
#include "DXSampleHelper.h"
#include "helper_cuda.h"

ComPtr<ID3D12Device4> Device::DX12Device;
ComPtr<IDXGIFactory5> Device::DXfactory;
cudaStream_t Device::Streams[FrameCount];
UINT Device::NodeMask;

void Device::InitDevice(HWND hWnd)
{
	UINT dxgiFactoryFlags = 0;
#if defined(_DEBUG)
	{
		ComPtr<ID3D12Debug> debugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
			debugController->EnableDebugLayer();
			dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
		}
	}
#endif
	ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&DXfactory)));
	DXfactory->MakeWindowAssociation(hWnd, DXGI_MWA_NO_ALT_ENTER);
	SIZE_T MaxSize = 0;
	ComPtr<IDXGIAdapter1> deviceAdapter;
	for (int idx = 0; DXfactory->EnumAdapters1(idx, &deviceAdapter) != DXGI_ERROR_NOT_FOUND; ++idx)
	{
		DXGI_ADAPTER_DESC1 desc;
		deviceAdapter->GetDesc1(&desc);
		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
		D3D12CreateDevice(deviceAdapter.Get(), D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr);
		wprintf(L"DX12 hardware:  %s (%u MB)\n", desc.Description, int(desc.DedicatedVideoMemory >> 20));
		if (desc.DedicatedVideoMemory > MaxSize) MaxSize = desc.DedicatedVideoMemory;
		
	}

	for (int idx = 0; DXfactory->EnumAdapters1(idx, &deviceAdapter) != DXGI_ERROR_NOT_FOUND; ++idx)
	{
		DXGI_ADAPTER_DESC1 desc;
		deviceAdapter->GetDesc1(&desc);
		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
		if (desc.DedicatedVideoMemory == MaxSize)
		{
			D3D12CreateDevice(deviceAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&DX12Device));
			wprintf(L"Used: DX12 hardware:  %s (%u MB)\n", desc.Description, int(desc.DedicatedVideoMemory >> 20));
			break;
		}
	}
	DXGI_ADAPTER_DESC1 desc;
	deviceAdapter->GetDesc1(&desc);


	int num_cuda_devices = 0;
	checkCudaErrors(cudaGetDeviceCount(&num_cuda_devices));

	if (!num_cuda_devices) {
		throw std::exception("No CUDA Devices found");
	}
	for (UINT devId = 0; devId < num_cuda_devices; devId++) {
		cudaDeviceProp devProp;
		checkCudaErrors(cudaGetDeviceProperties(&devProp, devId));

		if ((memcmp(&desc.AdapterLuid.LowPart, devProp.luid,
			sizeof(desc.AdapterLuid.LowPart)) == 0) &&
			(memcmp(&desc.AdapterLuid.HighPart,
				devProp.luid + sizeof(desc.AdapterLuid.LowPart),
				sizeof(desc.AdapterLuid.HighPart)) == 0)) {
			checkCudaErrors(cudaSetDevice(devId));
			
			NodeMask = devProp.luidDeviceNodeMask;
			for (int i = 0; i < FrameCount; i++) {
				checkCudaErrors(cudaStreamCreate(&Streams[i]));
			}
			printf("CUDA Device Used [%d] %s\n", devId, devProp.name);
			break;
		}
	}

}
