// GPUArrayLib.cu

// Project Includes
#include "GPUArrayLib.cuh"

// CUDA libaries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// C++ libaries
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>

// Image loading/saving libary
#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb/stb_image_write.h"

/*
* -------------------- KERNALS --------------------
*/

// Matrix - Scalar addition kernel
template <typename T>
__global__ void matrixScalarAdd(T* a, T b, T* c, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		c[absIdx] = a[absIdx] + b;
	}
}

// Matrix - Scalar subtraction kernel
template <typename T>
__global__ void matrixScalarSub(T* a, T b, T* c, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		c[absIdx] = a[absIdx] - b;
	}
}

// Matrix - Scalar multiplication kernel
template <typename T>
__global__ void matrixScalarMul(T* a, T b, T* c, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		c[absIdx] = a[absIdx] * b;
	}
}

// Matrix - Scalar division kernel
template <typename T>
__global__ void matrixScalarDiv(T* a, T b, T* c, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		c[absIdx] = a[absIdx] / b;
	}
}

// Matrix - Matrix addition kernel
template <typename T>
__global__ void matrixMatrixAdd(T* a, T* b, T* c, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		c[absIdx] = a[absIdx] + b[absIdx];
	}
}

// Matrix - Matrix subtraction kernel
template <typename T>
__global__ void matrixMatrixSub(T* a, T* b, T* c, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		c[absIdx] = a[absIdx] - b[absIdx];
	}
}

// Matrix - Matrix multiplication kernel
template <typename T>
__global__ void matrixMatrixMul(T* a, T* b, T* c, int NUM_BLOCKS, int dimZ, int a_dimY, int a_dimX, int b_dimY, int b_dimX, int c_dimY, int c_dimX) {
	// Allocate shared memory for the tiles of A and B
	__shared__ T A[SHMEM_SIZE];
	__shared__ T B[SHMEM_SIZE];

	int cx = blockIdx.x * blockDim.x + threadIdx.x; // Abs. X location (column) of the thread
	int cy = blockIdx.y * blockDim.y + threadIdx.y; // Abs. Y location (row) of the thread
	int cz = blockIdx.z * blockDim.z + threadIdx.z; // Abs. Z location (layer) of the thread

	// Check if result is within result matrix boundaries
	if (cz < dimZ && cy < c_dimY && cx < c_dimX) {

		T temp_sum = 0;

		// Loop over all tiles needed to compute the current value
		for (int i = 0; i < NUM_BLOCKS; i++) {
			// Load the current values of A and B into L1 cache
			if (cz < dimZ && cy < a_dimY && (i * blockDim.x + threadIdx.x) < a_dimX) {
				A[threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x] = a[cz * a_dimY * a_dimX + cy * a_dimX + (i * blockDim.x + threadIdx.x)];
			}
			if (cz < dimZ && (i * blockDim.y + threadIdx.y) < b_dimY && cx < b_dimX) {
				B[threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x] = b[cz * b_dimY * b_dimX + (i * blockDim.y + threadIdx.y) * b_dimX + cx];
			}

			// Wait for all threads to finish filling the tiles
			__syncthreads();

			// Compute the partial sum for the value in the current tile
			for (int k = 0; k < blockDim.x; k++) {
				if (cz < dimZ && (blockIdx.y * blockDim.y + k) < b_dimY && (blockIdx.x * blockDim.x + k) < a_dimX) {
					temp_sum += A[threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + k] * B[threadIdx.z * blockDim.y * blockDim.x + k * blockDim.x + threadIdx.x];
				}
			}

			// Wait for all threads to finish computing the partial sum
			__syncthreads();
		}
		// Store the final value in VRAM
		c[cz * c_dimY * c_dimX + cy * c_dimX + cx] = temp_sum;
	}
}

// Matrix - Matrix element-wise multiplication kernel
template <typename T1, typename T2>
__global__ void matrixElementMul(T1* a, T2* b, T1* c, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		c[absIdx] = a[absIdx] * b[absIdx];
	}
}

// Matrix - Matrix addition kernel with indexing
template <typename T>
__global__ void matrixAddIndexed(T* a, T* b, int* i, T* c, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		c[absIdx] = a[cz * dimY * dimX + cy * dimX + *(i + absIdx)] + b[cz * dimY * dimX + cy * dimX + *(i + absIdx)];
	}
}

// Kernal that rearanges the image data from RGBRGBRGB... to each channel in a seperate layer
__global__ void rearangeImageDataRGBtoLayer(unsigned char* RGBArray, unsigned char* layerArray, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		layerArray[cz * dimY * dimX + cy * dimX + cx] = RGBArray[cz + (3 * cy * dimX) + (3 * cx)];
	}
}

// Kernal that rearanges the image data from each channel in a seperate layer to RGBRGBRGB...
__global__ void rearangeImageDataLayertoRGB(unsigned char* layerArray, unsigned char* RGBArray, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		RGBArray[cz + (3 * cy * dimX) + (3 * cx)] = layerArray[cz * dimY * dimX + cy * dimX + cx];
	}
}

// Kernal that rearanges the flow data from each direction layer to RGBRGBRGB...
__global__ void rearangeFlowDataLayertoRGB(int* layerArray, unsigned char* RGBArray, int dimZ, int dimY, int dimX, int direction) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		RGBArray[cz + (3 * cy * dimX) + (3 * cx)] = (unsigned char)fminf(fmaxf(layerArray[direction * dimY * dimX + cy * dimX + cx] + 127, 0), 255);
	}
}

// Kernal that sets all array entries in the provided range to the provided value
template <typename T>
__global__ void setArrayEntriesAll(T* arrayPtrGPU, T value, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		arrayPtrGPU[absIdx] = value;
	}

}

// Kernal that sets all array entries in the provided range to the provided value
template <typename T>
__global__ void setArrayEntriesInRange(T* arrayPtrGPU, T value, int startIdx, int endIndex, int dimZ, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int absIdx = cz * dimY * dimX + cy * dimX + cx;

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		if (absIdx >= startIdx && absIdx <= endIndex) {
			arrayPtrGPU[absIdx] = value;
		}
	}
}

// Kernal that tests the VRAM of the GPU
__global__ void memTestKernal(long* arrayPtrGPU, long steps) {
	// Current entry to be computed by the thread
	long x = threadIdx.x;
	long idx = 0;

	for (idx = 0; idx < steps; idx++) {
		arrayPtrGPU[x * steps + idx] = idx;
	}
	for (idx = 0; idx < steps; idx++) {
		if (arrayPtrGPU[x * steps + idx] != idx) {
			printf("ERROR: %d != %d\n", arrayPtrGPU[x * steps + idx], idx);
		}
		else if ((idx % 10000000) == 0) {
			printf("T %d - %d MB / %d MB\n", x, idx / 1000000, (steps * 32) / 1000000);
		}
	}
}

/*
* -------------------- GPUArray CLASS --------------------
*/

/*
* Constructor for a standard multidimensional array
*
* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
* @param initializer: Value to initialize all array entries with
*
* @return: GPUArray object
*/
template <typename T>
GPUArray<T>::GPUArray(std::vector<int> arrayShape, T initializer) {
	// Set dimensions
	shape = arrayShape;
	dims = (int)arrayShape.size();
	if (dims == 0) {
		fprintf(stderr, "ERROR: No array dimensions given!\n");
		exit(-1);
	}
	else if (dims == 1) {
		dimX = 1;
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * sizeof(T);
	}
	else if (dims == 2) {
		dimX = *(arrayShape.end() - 1);
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * dimX * sizeof(T);
	}
	else {
		dimX = *(arrayShape.end() - 1);
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * dimX * sizeof(T);
		for (int i = 2; i < dims; i++) {
			dimZ *= *(arrayShape.end() - i - 1);
			bytes *= *(arrayShape.end() - i - 1);
		}
	}

	// Check if all dimensions are positive
	for (int i = 0; i < dims; i++) {
		if (*(arrayShape.begin() + i) <= 0) {
			fprintf(stderr, "ERROR: All array dimensions must be positive!\n");
			exit(-1);
		}
	}

	// Allocate host memory
	arrayPtrCPU = (T*)malloc(bytes);

	// Set all array entries to the initializer value
	for (int z = 0; z < dimZ; z++) {
		for (int y = 0; y < dimY; y++) {
			for (int x = 0; x < dimX; x++) {
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
* Constructor for an image array on the GPU
*
* @param filename: Path to the image file
* @param height: Height of the image
* @param width: Width of the image
*
*
* @return: GPUArray object
*/
GPUArray<unsigned char>::GPUArray(const char* filePath, int height, int width) {
	// Check if the image file exists
	FILE* file = fopen(filePath, "r");
	if (file == NULL) {
		std::cerr << "Error opening image file: " << filePath << std::endl;
		exit(-1);
	}

	// Check that width and height are positive
	if (width <= 0 || height <= 0) {
		std::cerr << "Error: Image dimensions must be positive!" << std::endl;
		exit(-1);
	}

	// Set dimensions
	shape = std::vector<int>{ 3, height, width };
	dims = 3;
	dimX = width;
	dimY = height;
	dimZ = 3;
	bytes = 3 * height * width;

	// Allocate host memory
	unsigned char* rawArrayPtrCPU = (unsigned char*)malloc(bytes);

	// Load image
	int channels = 3;
	rawArrayPtrCPU = stbi_load(filePath, &width, &height, &channels, STBI_rgb);

	// Check if the image was loaded successfully
	if (!rawArrayPtrCPU) {
		std::cerr << "Error loading image: " << filePath << std::endl;
		exit(-1);
	}

	// Allocate VRAM
	unsigned char* rawArrayPtrGPU;
	cudaMalloc(&rawArrayPtrGPU, bytes);

	// Copy host array to GPU
	cudaMemcpy(rawArrayPtrGPU, rawArrayPtrCPU, bytes, cudaMemcpyHostToDevice);

	// Free host memory
	stbi_image_free(rawArrayPtrCPU);

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 threads(NUM_THREADS, NUM_THREADS, 3);

	// Allocate VRAM
	cudaMalloc(&arrayPtrGPU, bytes);

	// Launch kernel
	rearangeImageDataRGBtoLayer << <grid, threads >> > (rawArrayPtrGPU, arrayPtrGPU, 3, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
	isOnGPU = true;

	// Free device memory
	cudaFree(rawArrayPtrGPU);
}

/*
* Constructor for an image batch array on the GPU
*
* @param filename: Path to the folder containing the image files
* @param batchSize: Number of images in the batch
* @param width: Width of the image
* @param height: Height of the image
*
* @return: GPUArray object
*/
GPUArray<unsigned char>::GPUArray(const char* filePath, int batchSize, int width, int height) {
	// Check that width and height are positive
	if (width < 1 || height < 1) {
		std::cerr << "Error: Image dimensions must be positive!" << std::endl;
		exit(-1);
	}

	// Set dimensions
	shape = std::vector<int>{ batchSize, 3, height, width };
	dims = 4;
	dimX = width;
	dimY = height;
	dimZ = batchSize * 3;
	bytes = batchSize * 3 * height * width;

	// Allocate host memory
	unsigned char* rawArrayPtrCPU = (unsigned char*)malloc(bytes);
	unsigned char* imageData = (unsigned char*)malloc(bytes / batchSize);

	// Load images
	int channels = 3;
	for (int i = 0; i < batchSize; i++) {
		// Format the file name
		std::ostringstream oss;
		oss << std::setw(5) << std::setfill('0') << i;
		std::string fileName = filePath + oss.str() + ".png";

		// Load the image
		imageData = stbi_load(fileName.c_str(), &width, &height, &channels, STBI_rgb);

		// Check if the image was loaded successfully
		if (!rawArrayPtrCPU) {
			std::cerr << "Error loading image: " << fileName << std::endl;
			exit(-1);
		}

		// Copy the loaded image data to the appropriate location in rawArrayPtrCPU
		memcpy(rawArrayPtrCPU + i * 3 * height * width, imageData, 3 * height * width);
	}
	stbi_image_free(imageData);

	// Allocate VRAM
	unsigned char* rawArrayPtrGPU;
	cudaMalloc(&rawArrayPtrGPU, bytes);

	// Copy host array to GPU
	cudaMemcpy(rawArrayPtrGPU, rawArrayPtrCPU, bytes, cudaMemcpyHostToDevice);

	// Free host memory
	stbi_image_free(rawArrayPtrCPU);

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Allocate VRAM
	cudaMalloc(&arrayPtrGPU, bytes);

	// Launch kernel
	rearangeImageDataRGBtoLayer << <grid, threads >> > (rawArrayPtrGPU, arrayPtrGPU, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
	isOnGPU = true;

	// Free device memory
	cudaFree(rawArrayPtrGPU);
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
		arrayPtrCPU = (T*)malloc(bytes);

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
void GPUArray<T>::download(T* memPointer) {
	// Copy host array to memory pointer
	cudaMemcpy(memPointer, arrayPtrGPU, bytes, cudaMemcpyDeviceToHost);
}

/*
* Changes the dimensions of the array
*
* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
*/
template <typename T>
void GPUArray<T>::changeDims(std::vector<int> arrayShape) {
	// Set dimensions
	shape = arrayShape;
	dims = (int)arrayShape.size();
	if (dims == 0) {
		fprintf(stderr, "ERROR: No array dimensions given!\n");
		exit(-1);
	}
	else if (dims == 1) {
		dimX = 1;
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * sizeof(T);
	}
	else if (dims == 2) {
		dimX = *(arrayShape.end() - 1);
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * dimX * sizeof(T);
	}
	else {
		dimX = *(arrayShape.end() - 1);
		dimY = *(arrayShape.end() - 2);
		dimZ = 1;
		bytes = dimY * dimX * sizeof(T);
		for (int i = 2; i < dims; i++) {
			dimZ *= *(arrayShape.end() - i - 1);
			bytes *= *(arrayShape.end() - i - 1);
		}
	}

	// Check if all dimensions are positive
	for (int i = 0; i < dims; i++) {
		if (*(arrayShape.begin() + i) <= 0) {
			fprintf(stderr, "ERROR: All array dimensions must be positive!\n");
			exit(-1);
		}
	}

	// Allocate host memory
	arrayPtrCPU = (T*)malloc(bytes);

	// Allocate VRAM
	cudaMalloc(&arrayPtrGPU, bytes);

	// Copy host array to GPU
	cudaMemcpy(arrayPtrGPU, arrayPtrCPU, bytes, cudaMemcpyHostToDevice);
	isOnGPU = true;

	// Free host memory
	free(arrayPtrCPU);
}

/*
* Adds a value to the array
*
* @param value: Value to add to the array
*/
template <typename T>
void GPUArray<T>::add(T value) {
	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Launch kernel
	matrixScalarAdd << <grid, threads >> > (arrayPtrGPU, value, arrayPtrGPU, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Subtracts a value from the array
*
* @param value: Value to subtract from the array
*/
template <typename T>
void GPUArray<T>::sub(T value) {
	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Launch kernel
	matrixScalarSub << <grid, threads >> > (arrayPtrGPU, value, arrayPtrGPU, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Scales the array by a scalar
*
* @param scalar: Scalar to scale the array by
*/
template <typename T>
void GPUArray<T>::mul(T scalar) {
	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Launch kernel
	matrixScalarMul << <grid, threads >> > (arrayPtrGPU, scalar, arrayPtrGPU, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Divides the array by a value
*
* @param value: Value to divide the array by
*/
template <typename T>
void GPUArray<T>::div(T value) {
	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Launch kernel
	matrixScalarDiv << <grid, threads >> > (arrayPtrGPU, value, arrayPtrGPU, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Adds another array to the array
*
* @param array: Array to add to the array
*/
template <typename T>
void GPUArray<T>::add(GPUArray<T>& array) {
	// Check if the matrices can be subtracted
	if (dimZ != array.dimZ || dimY != array.dimY || dimX != array.dimX) {
		fprintf(stderr, "ERROR: Array dimensions do not match!\n");
		exit(-1);
	}

	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}
	if (!array.isOnGPU) {
		array.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Launch kernel
	matrixMatrixAdd << <grid, threads >> > (arrayPtrGPU, array.arrayPtrGPU, arrayPtrGPU, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Subtracts another array from the array
*
* @param array: Array to subtract from the array
*/
template <typename T>
void GPUArray<T>::sub(GPUArray<T>& array) {
	// Check if the matrices can be subtracted
	if (dimZ != array.dimZ || dimY != array.dimY || dimX != array.dimX) {
		fprintf(stderr, "ERROR: Array dimensions do not match!\n");
		exit(-1);
	}

	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}
	if (!array.isOnGPU) {
		array.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Launch kernel
	matrixMatrixSub << <grid, threads >> > (arrayPtrGPU, array.arrayPtrGPU, arrayPtrGPU, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Multiplies another array to the array (this is not matrix multiplication)
*
* @param array: Array to multiply with the array
*/
template <typename T>
void GPUArray<T>::elementMul(GPUArray<T>& array) {
	// Check if the matrices can be subtracted
	if (dimZ != array.dimZ || dimY != array.dimY || dimX != array.dimX) {
		fprintf(stderr, "ERROR: Array dimensions do not match!\n");
		exit(-1);
	}

	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}
	if (!array.isOnGPU) {
		array.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Launch kernel
	matrixElementMul << <grid, threads >> > (arrayPtrGPU, array.arrayPtrGPU, arrayPtrGPU, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
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
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Set the array entries to the provided value
	setArrayEntriesAll << <grid, threads >> > (arrayPtrGPU, value, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
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
void GPUArray<T>::fill(T value, int startIdx, int endIndex) {
	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}

	// Check if the provided range is valid
	if (startIdx < 0 || endIndex >= dimZ * dimY * dimX) {
		fprintf(stderr, "ERROR: Provided range is invalid!\n");
		exit(-1);
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Set the array entries to the provided value
	setArrayEntriesInRange << <grid, threads >> > (arrayPtrGPU, value, startIdx, endIndex, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
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
void GPUArray<T>::fillData(T* memPointer) {
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
void GPUArray<T>::print<S>(int startIdx, int numElements) {
	// Check if the array is on the GPU
	if (isOnGPU) {
		toCPU();
	}

	int counter = 0;

	// Print the array
	for (int z = 0; z < dimZ; z++) {
		for (int y = 0; y < dimY; y++) {
			for (int x = 0; x < dimX; x++) {
				if (numElements == -1 || counter < numElements) {
					std::cout << (S) * (arrayPtrCPU + (startIdx + z * dimY * dimX + y * dimX + x)) << " ";
					counter++;
				}
				else {
					printf("\n");
					return;
				}
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

/*
* Returns the absolute index of an array entry given the indices for each dimension
*/
template <typename T>
int GPUArray<T>::getAbsIndex(std::initializer_list<int> indicesForEachDim) {
	// Check if index for every dim is provided
	int numIndices = (int)indicesForEachDim.size();
	if (numIndices != dims) {
		fprintf(stderr, "ERROR: Number of indexed dimensions does not match array dimensions!\n");
		exit(-1);
	}

	int absoluteIndex = 0; // Absolute index of the array entry (e.g. z * dimY * dimX + y * dimX + x)
	int subTerm = 0; // Current subterm (e.g. z * dimY * dimX)

	// Loop over the provided indices
	for (int currDim = 0; currDim < numIndices; currDim++) {
		subTerm = *(indicesForEachDim.begin() + currDim);

		// Check if the index is within the array bounds
		if ((subTerm < 0) || (subTerm >= *(shape.begin() + currDim))) {
			fprintf(stderr, "ERROR: One or more of the provided indices is outside of the array shape bounds!\n");
			exit(-1);
		}

		// Multiply the subterm with the array's dimensions (e.g. z * dimY * dimX)
		for (int mulDim = 1; mulDim < (numIndices - currDim); mulDim++) {
			subTerm *= *(shape.end() - mulDim);
		}

		// Add the subterm to the absolute index
		absoluteIndex += subTerm;
	}

	return absoluteIndex;
}

/*
* Exports the array as a PNG image
*
* @param filePath: Path to where the image should be saved
*/
template <typename T>
void GPUArray<T>::exportPNG(const char* filePath) {
	// Check if the array is compatible
	if (dims < 2) {
		fprintf(stderr, "ERROR: Array has too few dimensions to be exported as an image!\n");
		exit(-1);
	}
	else if (dims > 4) {
		fprintf(stderr, "ERROR: Array has too many dimensions to be exported as image(s)!\n");
		exit(-1);
	}

	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}

	// Allocate host memory
	unsigned char* rawArrayPtrCPU = (unsigned char*)malloc(bytes);

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);
	if (dims == 3) {
		NUM_BLOCKS_Z = 1;
	}

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);
	if (dims == 3) {
		dim3 threads(NUM_THREADS, NUM_THREADS, 3);
	}

	// Allocate VRAM
	unsigned char* rawArrayPtrGPU;
	cudaMalloc(&rawArrayPtrGPU, bytes);

	// Launch kernel
	rearangeImageDataLayertoRGB << <grid, threads >> > ((unsigned char*)arrayPtrGPU, rawArrayPtrGPU, dimZ, dimY, dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Copy host array to GPU
	cudaMemcpy(rawArrayPtrCPU, rawArrayPtrGPU, bytes, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(rawArrayPtrGPU);

	// Export single image
	if ((dimZ / 3) == 1) {
		stbi_write_png(filePath, dimX, dimY, 3, rawArrayPtrCPU, dimX * 3);
	}
	// Export image batch
	else {
		for (int i = 0; i < dimZ / 3; i++) {
			// Format the file name
			std::ostringstream oss;
			oss << std::setw(5) << std::setfill('0') << i;
			std::string fileName = filePath + oss.str() + ".png";

			// Export the image
			stbi_write_png(fileName.c_str(), dimX, dimY, 3, rawArrayPtrCPU + i * dimY * dimX * 3, dimX * 3);
		}
	}

	// Free host memory
	stbi_image_free(rawArrayPtrCPU);
}

/*
* Exports the array as a flow image
*
* @param filePath: Path to where the flow image should be saved
* @param direction: Direction of the flow (0: x, 1: y)
*/
void GPUArray<int>::exportFlowImage(const char* filePath, int direction) {
	// Check if the array is compatible
	if (dims < 2) {
		fprintf(stderr, "ERROR: Array has too few dimensions to be exported as an image!\n");
		exit(-1);
	}
	else if (dims > 4) {
		fprintf(stderr, "ERROR: Array has too many dimensions to be exported as image(s)!\n");
		exit(-1);
	}

	// Check if the array is on the GPU
	if (!isOnGPU) {
		toGPU();
	}

	// Allocate host memory
	size_t rgb_bytes = 3 * dimY * dimX;
	unsigned char* rawArrayPtrCPU = (unsigned char*)malloc(rgb_bytes);

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(dimZ / NUM_THREADS), 1);
	if (dims == 3) {
		NUM_BLOCKS_Z = 1;
	}

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);
	if (dims == 3) {
		dim3 threads(NUM_THREADS, NUM_THREADS, 3);
	}

	// Allocate VRAM
	unsigned char* rawArrayPtrGPU;
	cudaMalloc(&rawArrayPtrGPU, rgb_bytes);

	// Launch kernel
	rearangeFlowDataLayertoRGB << <grid, threads >> > (arrayPtrGPU, rawArrayPtrGPU, fmax(dimZ, 3), dimY, dimX, direction);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Copy host array to GPU
	cudaMemcpy(rawArrayPtrCPU, rawArrayPtrGPU, rgb_bytes, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(rawArrayPtrGPU);

	// Export single image
	if ((dimZ / 3) == 1) {
		stbi_write_png(filePath, dimX, dimY, 3, rawArrayPtrCPU, dimX * 3);
	}
	// Export image batch
	else {
		for (int i = 0; i < dimZ / 2; i++) {
			// Format the file name
			std::ostringstream oss;
			oss << std::setw(5) << std::setfill('0') << i;
			std::string fileName = filePath + oss.str() + ".png";

			// Export the image
			stbi_write_png(fileName.c_str(), dimX, dimY, 3, rawArrayPtrCPU + i * dimY * dimX * 3, dimX * 3);
		}
	}

	// Free host memory
	stbi_image_free(rawArrayPtrCPU);
}

/*
* Destructor
*/
template <typename T>
void GPUArray<T>::del() {
	if (isOnGPU) {
		cudaFree(arrayPtrGPU);
	}
	else {
		free(arrayPtrCPU);
	}
}



/*
* -------------------- MATRIX - VALUE OPERATIONS --------------------
*/



/*
* Adds a value to an array
*
* @param array: Array to be added to
* @param value: Value to add to the array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> add(GPUArray<T>& array, T value) {
	// Check if the matrix
	if (!array.isOnGPU) {
		array.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(array.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(array.dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(array.dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Initialize result array
	GPUArray<T> resultArray(array.shape, 0);

	// Launch kernel
	matrixScalarAdd << <grid, threads >> > (array.arrayPtrGPU, value, resultArray.arrayPtrGPU, array.dimZ, array.dimY, array.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return resultArray;
}

/*
* Subtracts a value from an array
*
* @param array: Array to be subtracted from
* @param value: Value to subtract from the array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> sub(GPUArray<T>& array, T value) {
	// Check if the matrix
	if (!array.isOnGPU) {
		array.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(array.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(array.dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(array.dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Initialize result array c
	GPUArray<T> resultArray(array.shape, 0);

	// Launch kernel
	matrixScalarSub << <grid, threads >> > (array.arrayPtrGPU, value, resultArray.arrayPtrGPU, array.dimZ, array.dimY, array.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return resultArray;
}

/*
* Scales an array by a scalar
*
* @param array: Array to be scaled
* @param value: Scalar to scale the array by
*
* @return: Result array
*/
template <typename T>
GPUArray<T> mul(GPUArray<T>& array, T scalar) {
	// Check if the matrix is on the GPU
	if (!array.isOnGPU) {
		array.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(array.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(array.dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(array.dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Initialize result array
	GPUArray<T> resultArray(array.shape, 0);

	// Launch kernel
	matrixScalarMul << <grid, threads >> > (array.arrayPtrGPU, scalar, resultArray.arrayPtrGPU, array.dimZ, array.dimY, array.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return resultArray;
}

/*
* Divides an array by a divisor value
*
* @param array: Array to be divided
* @param b: Divisor to divide the array by
*
* @return: Result array
*/
template <typename T>
GPUArray<T> div(GPUArray<T>& array, T value) {
	// Check if the matrix
	if (!array.isOnGPU) {
		array.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(array.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(array.dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(array.dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Initialize result array
	GPUArray<T> resultArray(array.shape, 0);

	// Launch kernel
	matrixScalarDiv << <grid, threads >> > (array.arrayPtrGPU, value, resultArray.arrayPtrGPU, array.dimZ, array.dimY, array.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return resultArray;
}



/*
* -------------------- MATRIX - MATRIX OPERATIONS --------------------
*/



/*
* Computes the sum of two arrays
*
* @param arrayA: First array
* @param arrayB: Second array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> add(GPUArray<T>& arrayA, GPUArray<T>& arrayB) {
	// Check if the matrices can be added
	if (arrayA.dimZ != arrayB.dimZ || arrayA.dimY != arrayB.dimY || arrayA.dimX != arrayB.dimX) {
		fprintf(stderr, "ERROR: Array dimensions do not match!\n");
		exit(-1);
	}

	// Check if the matrices are on the GPU
	if (!arrayA.isOnGPU) {
		arrayAtoGPU();
	}
	if (!arrayB.isOnGPU) {
		arrayB.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(arrayA.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(arrayA.dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(arrayA.dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Initialize result array
	GPUArray<T> resultArray(arrayA.shape, 0);

	// Launch kernel
	matrixMatrixAdd << <grid, threads >> > (arrayA.arrayPtrGPU, arrayB.arrayPtrGPU, resultArray.arrayPtrGPU, arrayA.dimZ, arrayA.dimY, arrayA.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return resultArray;
}

/*
* Subtracts two arrays
*
* @param arrayA: First array
* @param arrayB: Second array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> sub(GPUArray<T>& arrayA, GPUArray<T>& arrayB) {
	// Check if the matrices can be subtracted
	if (arrayA.dimZ != arrayB.dimZ || arrayA.dimY != arrayB.dimY || arrayA.dimX != arrayB.dimX) {
		fprintf(stderr, "ERROR: Array dimensions do not match!\n");
		exit(-1);
	}

	// Check if the matrices are on the GPU
	if (!arrayA.isOnGPU) {
		arrayA.toGPU();
	}
	if (!arrayB.isOnGPU) {
		arrayB.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(arrayA.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(arrayA.dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(arrayA.dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Initialize result array
	GPUArray<T> resultArray(arrayA.shape, 0);

	// Launch kernel
	matrixMatrixSub << <grid, threads >> > (arrayA.arrayPtrGPU, arrayB.arrayPtrGPU, resultArray.arrayPtrGPU, arrayA.dimZ, arrayA.dimY, arrayA.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return resultArray;
}

/*
* Computes the matrix multiplication of two arrays
*
* @param arrayA: First array
* @param arrayB: Second array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> mul(GPUArray<T>& arrayA, GPUArray<T>& arrayB) {
	// Check if the matrices can be multiplied
	if (arrayA.dimZ != arrayB.dimZ || arrayA.dimX != arrayB.dimY) {
		fprintf(stderr, "ERROR: Array dimensions do not match!\n");
		exit(-1);
	}

	// Check if the matrices are on the GPU
	if (!arrayA.isOnGPU) {
		arrayA.toGPU();
	}
	if (!arrayB.isOnGPU) {
		arrayB.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_XY = fmaxf(ceilf(arrayA.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(arrayA.dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_XY, NUM_BLOCKS_XY, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Initialize result array
	GPUArray<T> resultArray({ arrayA.dimZ, arrayA.dimY, arrayB.dimX }, 0);

	// Launch kernel
	matrixMatrixMul << <grid, threads >> > (arrayA.arrayPtrGPU, arrayB.arrayPtrGPU, resultArray.arrayPtrGPU, NUM_BLOCKS_XY, arrayA.dimZ, arrayA.dimY, arrayA.dimX, arrayB.dimY, arrayB.dimX, resultArray.dimY, resultArray.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return resultArray;
}

/*
* Multiplies two arrays together (this is not matrix multiplication)
*
* @param arrayA: First array
* @param arrayB: Second array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> elementMul(GPUArray<T>& arrayA, GPUArray<T>& arrayB) {
	// Check if the matrices can be subtracted
	if (arrayA.dimZ != arrayB.dimZ || arrayA.dimY != arrayB.dimY || arrayA.dimX != arrayB.dimX) {
		fprintf(stderr, "ERROR: Array dimensions do not match!\n");
		exit(-1);
	}

	// Check if the matrices are on the GPU
	if (!arrayA.isOnGPU) {
		arrayA.toGPU();
	}
	if (!arrayB.isOnGPU) {
		arrayB.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(arrayA.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(arrayA.dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(arrayA.dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Initialize result array
	GPUArray<T> resultArray(arrayA.shape, 0);

	// Launch kernel
	matrixElementMul << <grid, threads >> > (arrayA.arrayPtrGPU, arrayB.arrayPtrGPU, resultArray.arrayPtrGPU, arrayA.dimZ, arrayA.dimY, arrayA.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return resultArray;
}

/*
* Computes the array addition of two arrays at the provided indices
*
* @param arrayA: First array
* @param arrayB: Second array
* @param i: Indexing array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> addIndexed(GPUArray<T>& arrayA, GPUArray<T>& arrayB, GPUArray<int>& i) {
	// Check if the matrices can be added
	if (arrayA.dimZ != arrayB.dimZ || arrayA.dimY != arrayB.dimY || arrayA.dimX != arrayB.dimX || arrayA.dimZ != i.dimZ || arrayA.dimY != i.dimY || arrayA.dimX != i.dimX) {
		fprintf(stderr, "ERROR: Array dimensions do not match!\n");
		exit(-1);
	}

	// Check if the matrices are on the GPU
	if (!arrayA.isOnGPU) {
		arrayA.toGPU();
	}
	if (!arrayB.isOnGPU) {
		arrayB.toGPU();
	}
	if (!i.isOnGPU) {
		i.toGPU();
	}

	// Calculate the number of blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(arrayA.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(arrayA.dimY / NUM_THREADS), 1);
	int NUM_BLOCKS_Z = fmaxf(ceilf(arrayA.dimZ / NUM_THREADS), 1);

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);

	// Initialize result array
	GPUArray<T> resultArray(arrayA.shape, 0);

	// Launch kernel
	matrixAddIndexed << <grid, threads >> > (arrayA.arrayPtrGPU, arrayB.arrayPtrGPU, i.arrayPtrGPU, resultArray.arrayPtrGPU, arrayA.dimZ, arrayA.dimY, arrayA.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return resultArray;
}

/*
* Tests the memory of the GPU
*
* @param bytes: Number of bytes to allocate
*/
void memTest(size_t bytes) {
	// Allocate host memory
	long* arrayPtrCPU = (long*)malloc(bytes);

	// Allocate VRAM
	long* arrayPtrGPU;
	cudaMalloc(&arrayPtrGPU, bytes);

	// Copy host array to GPU
	cudaMemcpy(arrayPtrGPU, arrayPtrCPU, bytes, cudaMemcpyHostToDevice);

	// Free host memory
	free(arrayPtrCPU);

	// Use dim3 structs for block and grid size
	dim3 grid(1, 1, 1);
	dim3 threads(16, 1, 1);

	// Launch kernel
	memTestKernal << <grid, threads >> > (arrayPtrGPU, (bytes / 8) / 16);
	cudaDeviceSynchronize();

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Free device memory
	cudaFree(arrayPtrGPU);

	// Print the result
	printf("Memory test successful!\n");
}

// Explicit instantiations
template class GPUArray<int>;
template class GPUArray<unsigned int>;
template class GPUArray<float>;
template class GPUArray<double>;
template class GPUArray<long>;
template class GPUArray<unsigned char>;
template class GPUArray<bool>;

template void GPUArray<int>::print<int>(int startIdx, int numElements);
template void GPUArray<unsigned char>::print<int>(int startIdx, int numElements);
template void GPUArray<float>::print<float>(int startIdx, int numElements);
template void GPUArray<double>::print<double>(int startIdx, int numElements);
template void GPUArray<long>::print<long>(int startIdx, int numElements);
template void GPUArray<unsigned int>::print<unsigned int>(int startIdx, int numElements);
template void GPUArray<bool>::print<bool>(int startIdx, int numElements);