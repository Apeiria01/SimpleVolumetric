#pragma once
#include <Eigen/Core>
#include <texture_types.h>
#include "helper_cuda.h"

unsigned char* loadRawFile(const char* filename, size_t size);
//Should be pure CUDA texture, DX not involved.
template<typename DataType>
class VolumetricData {
private:
	cudaTextureObject_t m_deviceVolumetricTex;
	cudaTextureObject_t m_transferFunctionTex;
	cudaSurfaceObject_t m_gradient3DSurface;
	cudaArray_t m_deviceGradientArr;
	cudaArray_t m_deviceVolumetricArr;
	Eigen::Array3i m_size;
	Eigen::Array3f m_xyzPixelWidth;
	Eigen::Matrix4f m_worldToObjectMatrix;
	DataType* m_cpuVolumetricDataPtr;
	bool m_activated;
	static constexpr size_t m_dataTypeRawUnitSize = sizeof(DataType);
public:
	VolumetricData(const VolumetricData&) = delete;
	VolumetricData() = delete;
	VolumetricData(Eigen::Array3i size, Eigen::Array3f xyzPixelWidth);
	~VolumetricData() {
		if (m_cpuVolumetricDataPtr) delete[m_size[0] * m_size[1] * m_size[2]] m_cpuVolumetricDataPtr;
		checkCudaErrors(cudaDestroyTextureObject(m_deviceVolumetricTex));
		checkCudaErrors(cudaFreeArray(m_deviceVolumetricArr));
	}
	void Activate();
	DataType* GetCPUDataPtr() { return m_cpuVolumetricDataPtr; };
	cudaTextureObject_t GetGPUDataPtr() { return m_deviceVolumetricTex; };
	cudaTextureObject_t GetGPUTransferFuncPtr() { return m_transferFunctionTex; };
};


template<typename DataType>
VolumetricData<DataType>::VolumetricData(Eigen::Array3i size, Eigen::Array3f xyzPixelWidth) {
	m_size = size;
	m_xyzPixelWidth = xyzPixelWidth;
	m_cpuVolumetricDataPtr = new DataType[m_size[0] * m_size[1] * m_size[2]];
	m_activated = false;
	m_worldToObjectMatrix <<
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	;
}

template<typename DataType>
void VolumetricData<DataType>::Activate() {
	if (m_activated) return;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<DataType>();
	cudaExtent volumeSize = make_cudaExtent(m_size[0], m_size[1], m_size[2]);
	checkCudaErrors(cudaMalloc3DArray(&m_deviceVolumetricArr, &channelDesc, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr =
		make_cudaPitchedPtr(m_cpuVolumetricDataPtr, volumeSize.width * sizeof(DataType),
			volumeSize.width, volumeSize.height);
	copyParams.dstArray = m_deviceVolumetricArr;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));
	
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = m_deviceVolumetricArr;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true;  // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation
	texDescr.addressMode[0] = cudaAddressModeWrap;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeWrap;
	texDescr.addressMode[2] = cudaAddressModeWrap;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	checkCudaErrors(cudaCreateTextureObject(&m_deviceVolumetricTex, &texRes, &texDescr, nullptr));

	float4 transferFunc[] = {
	{  0.0, 0.0, 0.0, 0.0, },
	{  1.0, 0.0, 0.0, 1.0, },
	{  1.0, 0.5, 0.0, 1.0, },
	{  1.0, 1.0, 0.0, 1.0, },
	{  0.0, 1.0, 0.0, 1.0, },
	{  0.0, 1.0, 1.0, 1.0, },
	{  0.0, 0.0, 1.0, 1.0, },
	{  1.0, 0.0, 1.0, 1.0, },
	{  0.0, 0.0, 0.0, 0.0, },
	};

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	cudaArray* d_transferFuncArray;
	checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2,
		sizeof(transferFunc) / sizeof(float4), 1));
	checkCudaErrors(cudaMemcpy2DToArray(d_transferFuncArray, 0, 0, transferFunc,
		0, sizeof(transferFunc), 1,
		cudaMemcpyHostToDevice));
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_transferFuncArray;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords =
		true;  // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
	texDescr.readMode = cudaReadModeElementType;
	checkCudaErrors(cudaCreateTextureObject(&m_transferFunctionTex, &texRes, &texDescr, NULL));

	checkCudaErrors(cudaMalloc3DArray(&m_deviceGradientArr, &channelDesc2, volumeSize, cudaArraySurfaceLoadStore));//w*h*d, float4
	cudaResourceDesc gradRes;
	memset(&gradRes, 0, sizeof(cudaResourceDesc));

	gradRes.resType = cudaResourceTypeArray;
	gradRes.res.array.array = m_deviceVolumetricArr;
	cudaCreateSurfaceObject(&m_gradient3DSurface, &gradRes);
	m_activated = true;

}