// opticalFlowCalc.cu

// Project Includes
#include "opticalFlowCalc.cuh"

// C++ libaries
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <math.h>

// Kernal that calculates the absolute difference between two frames using the offset array
__global__ void calcImageDelta(unsigned char* frame1, unsigned char* frame2, unsigned char* imageDeltaArray, int* offsetArray, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = threadIdx.z;

	// Get the current offsets to use
	int offsetX = -offsetArray[cy * dimX + cx];
	int offsetY = -offsetArray[dimY * dimX + cy * dimX + cx];

	// Check if the thread is inside the frame (without offsets)
	if (cz < 3 && cy < dimY && cx < dimX) {
		// Current pixel is outside of frame
		if ((cy + offsetY < 0) || (cx + offsetX < 0) || (cy + offsetY > dimY) || (cx + offsetX > dimX)) {
			imageDeltaArray[cz * dimY * dimX + cy * dimX + cx] = 0;
		}
		// Current pixel is inside of frame
		else {
			imageDeltaArray[cz * dimY * dimX + cy * dimX + cx] = fabsf(frame1[cz * dimY * dimX + cy * dimX + cx + (offsetY * dimX + offsetX)] - frame2[cz * dimY * dimX + cy * dimX + cx]);
		}
	}
}

// Kernal that sums up all the pixel deltas of each window
__global__ void calcDeltaSums(unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray, int windowDimY, int windowDimX, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = threadIdx.z;
	int windowIndexX = cx / windowDimX;
	int windowIndexY = cy / windowDimY;

	// Check if the thread is inside the frame
	if (cz < 3 && cy < dimY && cx < dimX) {
		atomicAdd(&summedUpDeltaArray[(windowIndexY * windowDimY) * dimX + (windowIndexX * windowDimX)], imageDeltaArray[cz * dimY * dimX + cy * dimX + cx]);
	}
}

// Kernal that normalizes all the pixel deltas of each window
__global__ void normalizeDeltaSums(unsigned int* summedUpDeltaArray, float* normalizedDeltaArray, int* offsetArray, int windowDimY, int windowDimX, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the thread is a window represent
	if (cy % windowDimY == 0 && cx % windowDimX == 0) {
		// Get the current window information
		int offsetX = offsetArray[cy * dimX + cx];
		int offsetY = offsetArray[dimY * dimX + cy * dimX + cx];

		// Calculate the number of pixels in the window
		int numPixels = windowDimY * windowDimX;

		// Calculate the not overlapping pixels
		int numNotOverlappingPixels = 0;
		int overlapX = 0;
		int overlapY = 0;

		// Calculate the number of not overlapping pixels
		if (cx + windowDimX + fabsf(offsetX) > dimX) {
			overlapX = fabsf(offsetX);
		}
		else if (cx - offsetX < 0) {
			overlapX = fabsf(offsetX);
		}
		else {
			overlapX = 0;
		}

		if (cy + windowDimY + fabsf(offsetY) > dimY) {
			overlapY = fabsf(offsetY);
		}
		else if (cy - offsetY < 0) {
			overlapY = fabsf(offsetY);
		}
		else {
			overlapY = 0;
		}

		numNotOverlappingPixels = overlapY * overlapX;
		numPixels -= numNotOverlappingPixels;

		// Normalize the summed up delta
		normalizedDeltaArray[cy * dimX + cx] = (float)summedUpDeltaArray[cy * dimX + cx] / numPixels;
	}
}

// Kernal that compares two arrays to find the lowest values
__global__ void compareArrays(float* normalizedDeltaArrayOld, float* normalizedDeltaArrayNew, bool* isValueDecreasedArray, int windowDimY, int windowDimX, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the thread is a window represent
	if (cy % windowDimY == 0 && cx % windowDimX == 0) {
		// Compare the two values
		isValueDecreasedArray[cy * dimX + cx] = normalizedDeltaArrayNew[cy * dimX + cx] < normalizedDeltaArrayOld[cy * dimX + cx];
	}
}

// Kernal that adjusts the offset array based on the comparison results
__global__ void compositeOffsetArray(int* offsetArray, bool* isValueDecreasedArray, int* statusArray, int currentGlobalOffset, int windowDimY, int windowDimX, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int wx = (cx / windowDimX) * windowDimX;
	int wy = (cy / windowDimY) * windowDimY;

	/*
	* Status Array Key:
	* 0: Set the initial positive x direction
	* 1: Test the positive x direction
	* 2: Continue moving in the positive x direction
	* 3: Set the initial negative x direction
	* 4: Test the negative x direction
	* 5: Continue moving in the negative x direction
	* 6: Set the initial positive y direction
	* 7: Test the positive y direction
	* 8: Continue moving in the positive y direction
	* 9: Set the initial negative y direction
	* 10: Test the negative y direction
	* 11: Continue moving in the negative y direction
	* 12: Search complete
	*/

	if (cy < dimY && cx < dimX) {
		int currentStatus = statusArray[cy * dimX + cx];

		switch (currentStatus) {
			/*
			* X - DIRECTION
			*/
		case 0:
			// Set the initial positive x direction
			statusArray[cy * dimX + cx] = 1;
			offsetArray[cy * dimX + cx] += currentGlobalOffset;
			break;
		case 1:
			// Test the positive x direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				statusArray[cy * dimX + cx] = 2;
				offsetArray[cy * dimX + cx] += currentGlobalOffset;
			}
			else {
				statusArray[cy * dimX + cx] = 3;
				offsetArray[cy * dimX + cx] -= currentGlobalOffset;
			}
			break;
		case 2:
			// Continue moving in the positive x direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				offsetArray[cy * dimX + cx] += currentGlobalOffset;
			}
			else {
				statusArray[cy * dimX + cx] = 6;
				offsetArray[cy * dimX + cx] -= currentGlobalOffset;
			}
			break;
		case 3:
			// Set the initial negative x direction
			statusArray[cy * dimX + cx] = 4;
			offsetArray[cy * dimX + cx] -= currentGlobalOffset;
			break;
		case 4:
			// Test the negative x direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				statusArray[cy * dimX + cx] = 5;
				offsetArray[cy * dimX + cx] -= currentGlobalOffset;
			}
			else {
				statusArray[cy * dimX + cx] = 6;
				offsetArray[cy * dimX + cx] += currentGlobalOffset;
			}
			break;
		case 5:
			// Continue moving in the negative x direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				offsetArray[cy * dimX + cx] -= currentGlobalOffset;
			}
			else {
				statusArray[cy * dimX + cx] = 6;
				offsetArray[cy * dimX + cx] += currentGlobalOffset;
			}
			break;

			/*
			* Y - DIRECTION
			*/
		case 6:
			// Set the initial positive y direction
			statusArray[cy * dimX + cx] = 7;
			offsetArray[dimY * dimX + cy * dimX + cx] += currentGlobalOffset;
			break;
		case 7:
			// Test the positive y direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				statusArray[cy * dimX + cx] = 8;
				offsetArray[dimY * dimX + cy * dimX + cx] += currentGlobalOffset;
			}
			else {
				statusArray[cy * dimX + cx] = 9;
				offsetArray[dimY * dimX + cy * dimX + cx] -= currentGlobalOffset;
			}
			break;
		case 8:
			// Continue moving in the positive y direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				offsetArray[dimY * dimX + cy * dimX + cx] += currentGlobalOffset;
			}
			else {
				statusArray[cy * dimX + cx] = 12;
				offsetArray[dimY * dimX + cy * dimX + cx] -= currentGlobalOffset;
			}
			break;
		case 9:
			// Set the initial negative y direction
			statusArray[cy * dimX + cx] = 10;
			offsetArray[dimY * dimX + cy * dimX + cx] -= currentGlobalOffset;
			break;
		case 10:
			// Test the negative y direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				statusArray[cy * dimX + cx] = 11;
				offsetArray[dimY * dimX + cy * dimX + cx] -= currentGlobalOffset;
			}
			else {
				statusArray[cy * dimX + cx] = 12;
				offsetArray[dimY * dimX + cy * dimX + cx] += currentGlobalOffset;
			}
			break;
		case 11:
			// Continue moving in the negative y direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				offsetArray[dimY * dimX + cy * dimX + cx] -= currentGlobalOffset;
			}
			else {
				statusArray[cy * dimX + cx] = 12;
				offsetArray[dimY * dimX + cy * dimX + cx] += currentGlobalOffset;
			}
			break;
		case 12:
			// Search is complete
			break;
		default:
			break;
		}
	}
	return;
}

// Kernal that warps frame1 according to the offset array
__global__ void warpFrameKernal(unsigned char* frame1, int* offsetArray, int* hitCount, int* ones, unsigned char* warpedFrame, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = threadIdx.z;

	// Get the current offsets to use
	int offsetX = offsetArray[cy * dimX + cx];
	int offsetY = offsetArray[dimY * dimX + cy * dimX + cx];

	// Check if the thread is inside the frame (without offsets)
	if (cz < 3 && cy < dimY && cx < dimX) {
		// Check if the current pixel is inside of the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			warpedFrame[cz * dimY * dimX + cy * dimX + cx + (offsetY * dimX + offsetX)] = frame1[cz * dimY * dimX + cy * dimX + cx];
			atomicAdd(&hitCount[cz * dimY * dimX + cy * dimX + cx + (offsetY * dimX + offsetX)], ones[cz * dimY * dimX + cy * dimX + cx]);
		}
	}
}

// Kernal that removes artifacts from the warped frame
__global__ void artifactRemovalKernal(unsigned char* frame1, int* hitCount, unsigned char* warpedFrame, int dimY, int dimX) {
	// Current entry to be computed by the thread
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	int cz = threadIdx.z;

	// Check if the thread is inside the frame (without offsets)
	if (cz < 3 && cy < dimY && cx < dimX) {
		// Check if the current pixel is inside of the frame
		if (hitCount[cz * dimY * dimX + cy * dimX + cx] != 1) {
			warpedFrame[cz * dimY * dimX + cy * dimX + cx] = frame1[cz * dimY * dimX + cy * dimX + cx];
		}
	}
}

// Constructor
OpticalFlowCalc::OpticalFlowCalc()
{
}

/*
* Initializes the optical flow calculation
*
* @param dimY: The height of the frame
* @param dimX: The width of the frame
*/
void OpticalFlowCalc::init(int dimY, int dimX)
{
	NUM_BLOCKS_X = fmaxf(ceilf(dimX / NUM_THREADS), 1);
	NUM_BLOCKS_Y = fmaxf(ceilf(dimY / NUM_THREADS), 1);
	grid.x = NUM_BLOCKS_X;
	grid.y = NUM_BLOCKS_Y;
	grid.z = 1;
	threads3.x = NUM_THREADS;
	threads3.y = NUM_THREADS;
	threads3.z = 3;
	threads2.x = NUM_THREADS;
	threads2.y = NUM_THREADS;
	threads2.z = 2;
	threads1.x = NUM_THREADS;
	threads1.y = NUM_THREADS;
	threads1.z = 1;
	imageDeltaArray.changeDims({ 3, dimY, dimX });
	offsetArray.changeDims({ 2, dimY, dimX });
	statusArray.changeDims({ dimY, dimX });
	summedUpDeltaArray.changeDims({ dimY, dimX });
	normalizedDeltaArrayA.changeDims({ dimY, dimX });
	normalizedDeltaArrayB.changeDims({ dimY, dimX });
	isValueDecreasedArray.changeDims({ dimY, dimX });
	windowDimX = dimX;
	windowDimY = dimY;
	currentGlobalOffset = fmax(dimX / MAX_OFFSET_DIVIDER, 1);
	numIterations = ceil(log2f(dimX));
}

/*
* Calculates the optical flow between frame1 and frame2
*
* @param frame1: The frame to calculate the flow from
* @param frame2: The frame to calculate the flow to
*
* @return: The flow array containing the relative vectors
*/
GPUArray<unsigned char> OpticalFlowCalc::calculateOpticalFlow(GPUArray<unsigned char>& frame1, GPUArray<unsigned char>& frame2) {
	// Calculate the image deltas with the current offset array
	calcImageDelta << <grid, threads3 >> > (frame1.arrayPtrGPU, frame2.arrayPtrGPU, imageDeltaArray.arrayPtrGPU, offsetArray.arrayPtrGPU, frame1.dimY, frame1.dimX);

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return imageDeltaArray;
}

/*
* Warps frame1 according to the offset array
*
* @param frame1: The frame to warp
* @param offsetArray: The array containing the offsets
*
* @return: The warped frame
*/
GPUArray<unsigned char> OpticalFlowCalc::warpFrame(GPUArray<unsigned char>& frame1, GPUArray<int>& offsetArray) {
	if (DEBUG_MODE) {
		// Check if the dimensions match
		if (frame1.dimZ != 3 || offsetArray.dimZ != 3 || frame1.dimY != offsetArray.dimY || frame1.dimX != offsetArray.dimX) {
			fprintf(stderr, "ERROR: Frame and offset array dimensions do not match!\n");
			exit(-1);
		}

		// Check if the arrays are on the GPU
		if (!frame1.isOnGPU) {
			frame1.toGPU();
		}
		if (!offsetArray.isOnGPU) {
			offsetArray.toGPU();
		}
	}

	// Calculate the number of cuda blocks needed
	int NUM_BLOCKS_X = fmaxf(ceilf(frame1.dimX / NUM_THREADS), 1);
	int NUM_BLOCKS_Y = fmaxf(ceilf(frame1.dimY / NUM_THREADS), 1);

	// Calculate the number of cuda threads needed
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 threads3(NUM_THREADS, NUM_THREADS, 3);

	// Initialize result arrays
	GPUArray<unsigned char> warpedFrame(frame1.shape, 0); // Array containing the warped frame
	GPUArray<int> hitCount(frame1.shape, 0); // Array containing the number of times a pixel was hit
	GPUArray<int> ones(frame1.shape, 1); // Array containing only ones for atomic add

	auto start = std::chrono::high_resolution_clock::now();

	// Warp the frame
	warpFrameKernal << <grid, threads3 >> > (frame1.arrayPtrGPU, offsetArray.arrayPtrGPU, hitCount.arrayPtrGPU, ones.arrayPtrGPU, warpedFrame.arrayPtrGPU, frame1.dimY, frame1.dimX);

	// Wait for all threads to finish
	cudaDeviceSynchronize();

	// Remove artifacts
	artifactRemovalKernal << <grid, threads3 >> > (frame1.arrayPtrGPU, hitCount.arrayPtrGPU, warpedFrame.arrayPtrGPU, frame1.dimY, frame1.dimX);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<double, std::milli>(stop - start).count();
	std::cout << "\nWarp Calc Time: " << std::fixed << std::setprecision(4) << duration << " milliseconds" << std::endl;

	// Check for CUDA errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Return result array
	return warpedFrame;
}