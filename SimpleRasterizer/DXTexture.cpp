#include "DXTexture.h"
#include "d3dx12.h"
#include "Device.h"
#include "WindowsSecurityAttributes.h"
#include "helper_cuda.h"
using Device::DX12Device;
DXTexture::DXTexture(UINT width, UINT height, D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle, D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle) {
	m_width = width;
	m_height = height;
	auto inf = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32G32B32A32_FLOAT,
		width, height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
	ThrowIfFailed(DX12Device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_SHARED,
		&inf,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr,
		IID_PPV_ARGS(&m_textureBuffer)));
	D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
	SRVDesc = { DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING };
	SRVDesc.Texture2D.MipLevels = 1;
	Device::DX12Device->CreateShaderResourceView(m_textureBuffer.Get(), &SRVDesc, cpuHandle);
	m_CPUDescriptor = cpuHandle;
	m_GPUDescriptor = gpuHandle;


    HANDLE sharedHandle;
    WindowsSecurityAttributes windowsSecurityAttributes;
    LPCWSTR name = NULL;

    ThrowIfFailed(Device::DX12Device->CreateSharedHandle(
        m_textureBuffer.Get(), &windowsSecurityAttributes, GENERIC_ALL, name,
        &sharedHandle));


    D3D12_RESOURCE_ALLOCATION_INFO d3d12TexAllocationInfo;
    d3d12TexAllocationInfo = Device::DX12Device->GetResourceAllocationInfo(
        Device::NodeMask, 1, &inf);
    size_t texSize = d3d12TexAllocationInfo.SizeInBytes;

    cudaExternalMemoryHandleDesc cuExtmemHandleDesc{};
    cuExtmemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    cuExtmemHandleDesc.handle.win32.handle = sharedHandle;
    cuExtmemHandleDesc.size = d3d12TexAllocationInfo.SizeInBytes;
    cuExtmemHandleDesc.flags = cudaExternalMemoryDedicated;

    checkCudaErrors(cudaImportExternalMemory(&m_externalMemory, &cuExtmemHandleDesc));
    CloseHandle(sharedHandle);

    cudaMipmappedArray_t cuMipArray{};
    cudaExternalMemoryMipmappedArrayDesc externalMemoryTextureDesc{};
    externalMemoryTextureDesc.extent = make_cudaExtent(inf.Width, inf.Height, 0);
    externalMemoryTextureDesc.formatDesc = cudaCreateChannelDesc<float4>();
    externalMemoryTextureDesc.numLevels = 1;
    externalMemoryTextureDesc.flags = cudaArraySurfaceLoadStore;
    checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&cuMipArray, m_externalMemory, &externalMemoryTextureDesc));

    cudaArray_t cuArray{};
    checkCudaErrors(cudaGetMipmappedArrayLevel(&cuArray, cuMipArray, 0));

    cudaResourceDesc cuResDesc{};
    cuResDesc.resType = cudaResourceTypeArray;
    cuResDesc.res.array.array = cuArray;
    checkCudaErrors(cudaCreateSurfaceObject(&m_DXTexturePTR, &cuResDesc));
}

DXTexture::~DXTexture() {
    checkCudaErrors(cudaDestroyExternalMemory(m_externalMemory));
    checkCudaErrors(cudaDestroySurfaceObject(m_DXTexturePTR));
    
}

