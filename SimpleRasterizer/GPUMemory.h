#pragma once

#include "helper_cuda.h"
#include "cuda_runtime.h"
#include <vector>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <atomic>
#include <string>
#include <stdexcept>

inline std::atomic<size_t>& total_n_bytes_allocated() {
	static std::atomic<size_t> s_total_n_bytes_allocated{ 0 };
	return s_total_n_bytes_allocated;
}

template<class T>
class GPUMemory {
private:
	T* m_data = nullptr;
	size_t m_size = 0; // Number of elements

public:
	GPUMemory() {}

	GPUMemory<T>& operator=(GPUMemory<T>&& other) {
		std::swap(m_data, other.m_data);
		std::swap(m_size, other.m_size);
		return *this;
	}

	GPUMemory(GPUMemory<T>&& other) {
		*this = std::move(other);
	}

	explicit GPUMemory(const GPUMemory<T>& other) {
		copy_from_device(other);
	}


	void allocate_memory(size_t n_bytes) {
		if (n_bytes == 0) {
			return;
		}
		uint8_t* rawptr = nullptr;
		(cudaMalloc(&rawptr, n_bytes));
		m_data = (T*)(rawptr);
		total_n_bytes_allocated() += n_bytes;
	}

	void free_memory() {
		if (!m_data) {
			return;
		}

		uint8_t* rawptr = (uint8_t*)m_data;
		checkCudaErrors(cudaFree(rawptr));

		total_n_bytes_allocated() -= get_bytes();

		m_data = nullptr;
	}

	GPUMemory(const size_t size) {
		allocate_memory(size);
	}

	/// Frees memory again
	__host__ __device__ ~GPUMemory() {
#ifndef __CUDA_ARCH__
		try {
			if (m_data) {
				free_memory();
				m_size = 0;
			}
		}
		catch (std::runtime_error error) {
			if (std::string{ error.what() }.find("driver shutting down") == std::string::npos) {
				fprintf(stderr, "Could not free memory: %s\n", error.what());
			}
		}
#endif
	}

	/// Sets the memory of the first num_elements to value
	void memset(const int value, const size_t num_elements, const size_t offset = 0) {
		if (num_elements + offset > m_size) {
			throw std::runtime_error("Could not set memory: Number of elements larger than allocated memory");
		}

		try {
			checkCudaErrors(cudaMemset(m_data + offset, value, num_elements * sizeof(T)));
		}
		catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not set memory: ") + error.what());
		}
	}

	/// Sets the memory of the all elements to value
	void memset(const int value) {
		memset(value, m_size);
	}
	/** @} */

	/** @name Copy operations
	 *  @{
	 */
	 /// Copy data of num_elements from the raw pointer on the host
	void copy_from_host(const T* host_data, const size_t num_elements) {
		try {
			checkCudaErrors(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
		}
		catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from host: ") + error.what());
		}
	}

	/// Copy num_elements from the host vector
	void copy_from_host(const std::vector<T>& data, const size_t num_elements) {
		if (data.size() < num_elements) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(num_elements) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_from_host(data.data(), num_elements);
	}

	/// Copies data from the raw host pointer to fill the entire array
	void copy_from_host(const T* data) {
		copy_from_host(data, m_size);
	}

	/// Copies the entire host vector to the device. Fails if there is not enough space available.
	void copy_from_host(const std::vector<T>& data) {
		if (data.size() < m_size) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(m_size) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_from_host(data.data(), m_size);
	}

	/// Copies num_elements of data from the raw host pointer to the device. Fails if there is not enough space available.
	void copy_to_host(T* host_data, const size_t num_elements) const {
		if (num_elements > m_size) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(num_elements) + std::string(" elements, but vector size is only ") + std::to_string(m_size));
		}
		try {
			checkCudaErrors(cudaMemcpy(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));
		}
		catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy to host: ") + error.what());
		}
	}

	void copy_to_host(std::vector<T>& data, const size_t num_elements) const {
		if (data.size() < num_elements) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(num_elements) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_to_host(data.data(), num_elements);
	}

	/// Copies num_elements from the device to a raw pointer on the host
	void copy_to_host(T* data) const {
		copy_to_host(data, m_size);
	}

	/// Copies all elements from the device to a vector on the host
	void copy_to_host(std::vector<T>& data) const {
		if (data.size() < m_size) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(m_size) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_to_host(data.data(), m_size);
	}

	/// Copies size elements from another device array to this one, automatically resizing it
	void copy_from_device(const GPUMemory<T>& other, const size_t size) {
		if (size == 0) {
			return;
		}
		if (m_size < size) {
			return;
		}
		try {
			checkCudaErrors(cudaMemcpy(m_data, other.m_data, size * sizeof(T), cudaMemcpyDeviceToDevice));
		}
		catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from device: ") + error.what());
		}
	}

	/// Copies data from another device array to this one, automatically resizing it
	void copy_from_device(const GPUMemory<T>& other) {
		copy_from_device(other, other.m_size);
	}

	// Created an (owned) copy of the data
	GPUMemory<T> copy(size_t size) const {
		GPUMemory<T> result{ size };
		result.copy_from_device(*this);
		return result;
	}

	GPUMemory<T> copy() const {
		return copy(m_size);
	}

	__host__ __device__ T* data() const {
		return m_data;
	}

	__host__ __device__ T& operator[](size_t idx) const {
		return m_data[idx];
	}

	__host__ __device__ T& operator[](uint32_t idx) const {
		return m_data[idx];
	}

	size_t get_num_elements() const {
		return m_size;
	}

	size_t size() const {
		return get_num_elements();
	}

	size_t get_bytes() const {
		return m_size * sizeof(T);
	}

	size_t bytes() const {
		return get_bytes();
	}
};