#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUArrayLib.cuh"

#include <amvideo.h>

/*
* -------------------- KERNELS --------------------
*/

// Kernel that sets all array entries in the provided range to the provided value
template <typename T>
__global__ void setArrayEntriesAll(T* arrayPtrGPU, T value, const unsigned int dimZ, const unsigned int dimY,
	const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;
	const unsigned int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		arrayPtrGPU[absIdx] = value;
	}
}

// Kernel that sets all array entries in the provided range to the provided value
template <typename T>
__global__ void setArrayEntriesInRange(T* arrayPtrGPU, T value, const unsigned int startIdx,
	const unsigned int endIndex, const unsigned int dimZ, const unsigned int dimY,
	const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;
	const unsigned int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		if (absIdx >= startIdx && absIdx <= endIndex) {
			arrayPtrGPU[absIdx] = value;
		}
	}
}

/*
* -------------------- GPUArray CLASS --------------------
*/

/*
* Default constructor for a standard multidimensional array
*
* @return: GPUArray object
*/
template <typename T>
GPUArray<T>::GPUArray() {
	arrayPtrCPU = nullptr;
	arrayPtrGPU = nullptr;
	dims = 0;
	std::vector<unsigned int> shape{0};
	dimX = 0;
	dimY = 0;
	dimZ = 0;
	bytes = 0;
	isOnGPU = false;
}

/*
* Constructor for a standard multidimensional array
*
* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
* @param initializer: Value to initialize all array entries with
*
* @return: GPUArray object
*/
template <typename T>
GPUArray<T>::GPUArray(std::vector<unsigned int> arrayShape, T initializer) {
	// Set dimensions
	shape = arrayShape;
	dims = static_cast<int>(arrayShape.size());
	if (dims == 0) {
		fprintf(stderr, "ERROR: No array dimensions given!\n");
		exit(-1);
	}
	if (dims == 1) {
		dimX = 1;
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * sizeof(T);
	} else if (dims == 2) {
		dimX = *(arrayShape.end() - 1);
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * dimX * sizeof(T);
	} else {
		dimX = *(arrayShape.end() - 1);
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * dimX * sizeof(T);
		for (unsigned int i = 2; i < dims; i++) {
			dimZ *= *(arrayShape.end() - i - 1);
			bytes *= *(arrayShape.end() - i - 1);
		}
	}

	// Check if all dimensions are positive
	for (unsigned int i = 0; i < dims; i++) {
		if (*(arrayShape.begin() + i) <= 0) {
			fprintf(stderr, "ERROR: All array dimensions must be positive!\n");
			exit(-1);
		}
	}

	// Allocate host memory
	arrayPtrCPU = static_cast<T*>(malloc(bytes));

	// Set all array entries to the initializer value
	for (unsigned int z = 0; z < dimZ; z++) {
		for (unsigned int y = 0; y < dimY; y++) {
			for (unsigned int x = 0; x < dimX; x++) {
				*(arrayPtrCPU + z * dimY * dimX + y * dimX + x) = initializer;
			}
		}
	}

	// Allocate VRAM
	cudaMalloc(&arrayPtrGPU, bytes);

	// Copy host array to GPU
	cudaMemcpy(arrayPtrGPU, arrayPtrCPU, bytes, cudaMemcpyHostToDevice);
	isOnGPU = true;

	// Free host memory
	free(arrayPtrCPU);
}

/*
* Destructor
*/
template <typename T>
GPUArray<T>::~GPUArray() {
	if (isOnGPU && arrayPtrGPU != nullptr) {
		cudaFree(arrayPtrGPU);
	} else if (!isOnGPU && arrayPtrCPU != nullptr) {
		free(arrayPtrCPU);
	}
}

/*
* Transfers the array to the GPU
*/
template <typename T>
void GPUArray<T>::toGPU() {
	if (!isOnGPU) {
		// Allocate VRAM
		cudaMalloc(&arrayPtrGPU, bytes);

		// Copy host array to GPU
		cudaMemcpy(arrayPtrGPU, arrayPtrCPU, bytes, cudaMemcpyHostToDevice);
		isOnGPU = true;

		// Free host memory
		free(arrayPtrCPU);
	}
}

/*
* Transfers the array to the CPU
*/
template <typename T>
void GPUArray<T>::toCPU() {
	if (isOnGPU) {
		// Allocate host memory
		arrayPtrCPU = static_cast<T*>(malloc(bytes));

		// Copy host array to GPU
		cudaMemcpy(arrayPtrCPU, arrayPtrGPU, bytes, cudaMemcpyDeviceToHost);
		isOnGPU = false;

		// Free device memory
		cudaFree(arrayPtrGPU);
	}
}

/*
* Transfers the array to the provided memory pointer
*
* @param memPointer: Pointer to the memory to transfer the array to
*/
template <typename T>
void GPUArray<T>::download(unsigned char* memPointer) const {
	// Copy host array to memory pointer
	cudaMemcpy(memPointer, arrayPtrGPU, bytes, cudaMemcpyDeviceToHost);
}

/*
* Changes the dimensions of the array
*
* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
* @param initializer: Value to initialize all array entries with
* @param bytesRequest: Size of the array in bytes (optional)
*/
template <typename T>
void GPUArray<T>::changeDims(std::vector<unsigned int> arrayShape, T initializer, const size_t bytesRequest) {
	// Set dimensions
	shape = arrayShape;
	dims = static_cast<int>(arrayShape.size());
	if (dims == 0) {
		fprintf(stderr, "ERROR: No array dimensions given!\n");
		exit(-1);
	}
	if (dims == 1) {
		dimX = 1;
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * sizeof(T);
	} else if (dims == 2) {
		dimX = *(arrayShape.end() - 1);
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * dimX * sizeof(T);
	} else {
		dimX = *(arrayShape.end() - 1);
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * dimX * sizeof(T);
		for (unsigned int i = 2; i < dims; i++) {
			dimZ *= *(arrayShape.end() - i - 1);
			bytes *= *(arrayShape.end() - i - 1);
		}
	}

	// Account for the provided bytes request
	if (bytesRequest > 0) {
		bytes = bytesRequest;
	}

	// Check if all dimensions are positive
	for (unsigned int i = 0; i < dims; i++) {
		if (*(arrayShape.begin() + i) <= 0) {
			fprintf(stderr, "ERROR: All array dimensions must be positive!\n");
			exit(-1);
		}
	}

	// Allocate host memory
	arrayPtrCPU = static_cast<T*>(malloc(bytes));

	// Set all array entries to 0
	if (bytesRequest > 0) {
		for (int i = 0; i < bytes / sizeof(T); i++) {
			*(arrayPtrCPU + i) = initializer;
		}
	} else {
		// Set all array entries to the initializer value
		for (unsigned int z = 0; z < dimZ; z++) {
			for (unsigned int y = 0; y < dimY; y++) {
				for (unsigned int x = 0; x < dimX; x++) {
					*(arrayPtrCPU + z * dimY * dimX + y * dimX + x) = initializer;
				}
			}
		}
	}

	// Allocate VRAM
	cudaMalloc(&arrayPtrGPU, bytes);

	// Copy host array to GPU
	cudaMemcpy(arrayPtrGPU, arrayPtrCPU, bytes, cudaMemcpyHostToDevice);
	isOnGPU = true;

	// Free host memory
	free(arrayPtrCPU);
}

/*
* Returns whether the array is initialized (i.e. has allocated memory)
*
* @return: True if the array is initialized, false otherwise
*/
template <typename T>
bool GPUArray<T>::isInitialized() const {
	return bytes;
}

/*
* Sets every array entry to 0
*/
template <typename T>
void GPUArray<T>::zero() const {
	cudaMemset(arrayPtrGPU, 0, bytes);
}

/*
* Sets every array entry to the provided value
*
* @param value: Value to set the array entries to
*/
template <typename T>
void GPUArray<T>::fill(T value) {
	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}

	// Calculate the number of blocks needed
	const int NUM_BLOCKS_X = static_cast<int>(fmaxf(ceilf(dimX / static_cast<float>(8)), 1));
	const int NUM_BLOCKS_Y = static_cast<int>(fmaxf(ceilf(dimY / static_cast<float>(8)), 1));
	const int NUM_BLOCKS_Z = static_cast<int>(fmaxf(ceilf(dimZ / static_cast<float>(8)), 1));

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(8, 8, 8);

	// Set the array entries to the provided value
	setArrayEntriesAll << <grid, threads >> > (arrayPtrGPU, value, dimZ, dimY, dimX);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Sets every array entry in the provided range to the provided value
*
* @param value: Value to set the array entries to
* @param startIdx: Index of the first array entry to set
* @param endIndex: Index of the last array entry to set
*/
template <typename T>
void GPUArray<T>::fill(T value, unsigned int startIdx, unsigned int endIndex) {
	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}

	// Check if the provided range is valid
	if (endIndex >= dimZ * dimY * dimX) {
		fprintf(stderr, "ERROR: Provided range is invalid!\n");
		exit(-1);
	}

	// Calculate the number of blocks needed
	const int NUM_BLOCKS_X = static_cast<int>(fmaxf(ceilf(dimX / static_cast<float>(8)), 1));
	const int NUM_BLOCKS_Y = static_cast<int>(fmaxf(ceilf(dimY / static_cast<float>(8)), 1));
	const int NUM_BLOCKS_Z = static_cast<int>(fmaxf(ceilf(dimZ / static_cast<float>(8)), 1));

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(8, 8, 8);

	// Set the array entries to the provided value
	setArrayEntriesInRange << <grid, threads >> > (arrayPtrGPU, value, startIdx, endIndex, dimZ, dimY, dimX);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR1: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Fills the array with data from system memory
*
* @param memPointer: Pointer to the data in system memory
* @param bytes: Size of the data in bytes
*/
template <typename T>
void GPUArray<T>::fillData(const unsigned char* memPointer) const {
	// Copy the data to the GPU
	cudaMemcpy(arrayPtrGPU, memPointer, bytes, cudaMemcpyHostToDevice);
}

/*
* Prints the elements of the array
*
* @param startIdx: Index of the first array entry to print
* @param numElements: How many array entries to print (-1 : all will be printed)
*/
template <typename T>
template <typename S>
void GPUArray<T>::print<S>(const unsigned int startIdx, const int numElements) {
	// Check if the array is on the GPU
	if (isOnGPU) {
		toCPU();
	}

	int counter = 0;

	// Print the array
	for (unsigned int z = 0; z < dimZ; z++) {
		for (unsigned int y = 0; y < dimY; y++) {
			for (unsigned int x = 0; x < dimX; x++) {
				if (numElements == -1 || counter < numElements) {
					std::cout << static_cast<S>(*(arrayPtrCPU + (startIdx + z * dimY * dimX + y * dimX + x))) << " ";
					counter++;
				} else {
					printf("\n");
					return;
				}
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");

	toGPU();
}

// Explicit instantiations
template class GPUArray<int>;
template class GPUArray<unsigned int>;
template class GPUArray<unsigned short>;
template class GPUArray<float>;
template class GPUArray<double>;
template class GPUArray<long>;
template class GPUArray<unsigned char>;
template class GPUArray<bool>;

template void GPUArray<int>::print<int>(unsigned int startIdx, int numElements);
template void GPUArray<unsigned char>::print<int>(unsigned int startIdx, int numElements);
template void GPUArray<float>::print<float>(unsigned int startIdx, int numElements);
template void GPUArray<double>::print<double>(unsigned int startIdx, int numElements);
template void GPUArray<long>::print<long>(unsigned int startIdx, int numElements);
template void GPUArray<unsigned int>::print<unsigned int>(unsigned int startIdx, int numElements);
template void GPUArray<bool>::print<bool>(unsigned int startIdx, int numElements);