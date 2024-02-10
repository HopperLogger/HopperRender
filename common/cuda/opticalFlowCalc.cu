// Project Includes
#include "opticalFlowCalc.cuh"

// C++ libraries
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>

// Kernel that calculates the absolute difference between two frames using the offset array
__global__ void calcImageDelta(const unsigned char* frame1, const unsigned char* frame2, unsigned char* imageDeltaArray, const int* offsetArray, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Get the current offsets to use
	const int offsetX = -offsetArray[cy * dimX + cx];
	const int offsetY = -offsetArray[dimY * dimX + cy * dimX + cx];

	// Check if the thread is inside the frame (without offsets)
	if (cz < 3 && cy < dimY && cx < dimX) {
		// Current pixel is outside of frame
		if ((cy + offsetY < 0) || (cx + offsetX < 0) || (cy + offsetY > dimY) || (cx + offsetX > dimX)) {
			imageDeltaArray[cz * dimY * dimX + cy * dimX + cx] = 0;
		// Current pixel is inside of frame
		} else {
			imageDeltaArray[cz * dimY * dimX + cy * dimX + cx] = fabsf(frame1[cz * dimY * dimX + cy * dimX + cx + (offsetY * dimX + offsetX)] - frame2[cz * dimY * dimX + cy * dimX + cx]);
		}
	}
}

// Kernel that sums up all the pixel deltas of each window
__global__ void calcDeltaSums(unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int windowDimY, const unsigned int windowDimX, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int windowIndexX = cx / windowDimX;
	const unsigned int windowIndexY = cy / windowDimY;

	// Check if the thread is inside the frame
	if (cz < 3 && cy < dimY && cx < dimX) {
		atomicAdd(&summedUpDeltaArray[(windowIndexY * windowDimY) * dimX + (windowIndexX * windowDimX)], imageDeltaArray[cz * dimY * dimX + cy * dimX + cx]);
	}
}

// Kernel that normalizes all the pixel deltas of each window
__global__ void normalizeDeltaSums(const unsigned int* summedUpDeltaArray, float* normalizedDeltaArray, const int* offsetArray, const unsigned int windowDimY, const unsigned int windowDimX, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the thread is a window represent
	if (cy % windowDimY == 0 && cx % windowDimX == 0) {
		// Get the current window information
		const int offsetX = offsetArray[cy * dimX + cx];
		const int offsetY = offsetArray[dimY * dimX + cy * dimX + cx];

		// Calculate the number of pixels in the window
		unsigned int numPixels = windowDimY * windowDimX;

		// Calculate the not overlapping pixels
		int numNotOverlappingPixels = 0;
		int overlapX = 0;
		int overlapY = 0;

		// Calculate the number of not overlapping pixels
		if ((cx + windowDimX + fabsf(offsetX) > dimX) || (cx - offsetX < 0)) {
			overlapX = fabsf(offsetX);
		} else {
			overlapX = 0;
		}

		if ((cy + windowDimY + fabsf(offsetY) > dimY) || (cy - offsetY < 0)) {
			overlapY = fabsf(offsetY);
		} else {
			overlapY = 0;
		}

		numNotOverlappingPixels = overlapY * overlapX;
		numPixels -= numNotOverlappingPixels;

		// Normalize the summed up delta
		normalizedDeltaArray[cy * dimX + cx] = (float)summedUpDeltaArray[cy * dimX + cx] / numPixels;
	}
}

// Kernel that compares two arrays to find the lowest values
__global__ void compareArrays(const float* normalizedDeltaArrayOld, const float* normalizedDeltaArrayNew, bool* isValueDecreasedArray, const unsigned int windowDimY, const unsigned int windowDimX, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the thread is a window represent
	if (cy % windowDimY == 0 && cx % windowDimX == 0) {
		// Compare the two values
		isValueDecreasedArray[cy * dimX + cx] = normalizedDeltaArrayNew[cy * dimX + cx] < normalizedDeltaArrayOld[cy * dimX + cx];
	}
}

// Kernel that adjusts the offset array based on the comparison results
__global__ void compositeOffsetArray(int* offsetArray, const bool* isValueDecreasedArray, int* statusArray, const int currentGlobalOffset, const unsigned int windowDimY, const unsigned int windowDimX, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int wx = (cx / windowDimX) * windowDimX;
	const unsigned int wy = (cy / windowDimY) * windowDimY;

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
		const int currentStatus = statusArray[cy * dimX + cx];

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
			} else {
				statusArray[cy * dimX + cx] = 3;
				offsetArray[cy * dimX + cx] -= currentGlobalOffset;
			}
			break;
		case 2:
			// Continue moving in the positive x direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				offsetArray[cy * dimX + cx] += currentGlobalOffset;
			} else {
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
			} else {
				statusArray[cy * dimX + cx] = 6;
				offsetArray[cy * dimX + cx] += currentGlobalOffset;
			}
			break;
		case 5:
			// Continue moving in the negative x direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				offsetArray[cy * dimX + cx] -= currentGlobalOffset;
			} else {
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
			} else {
				statusArray[cy * dimX + cx] = 9;
				offsetArray[dimY * dimX + cy * dimX + cx] -= currentGlobalOffset;
			}
			break;
		case 8:
			// Continue moving in the positive y direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				offsetArray[dimY * dimX + cy * dimX + cx] += currentGlobalOffset;
			} else {
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
			} else {
				statusArray[cy * dimX + cx] = 12;
				offsetArray[dimY * dimX + cy * dimX + cx] += currentGlobalOffset;
			}
			break;
		case 11:
			// Continue moving in the negative y direction
			if (isValueDecreasedArray[wy * dimX + wx]) {
				offsetArray[dimY * dimX + cy * dimX + cx] -= currentGlobalOffset;
			} else {
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
}

// Kernel that warps frame1 according to the offset array
__global__ void warpFrameKernel(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones, unsigned char* warpedFrame, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Get the current offsets to use
	const int offsetX = offsetArray[cy * dimX + cx];
	const int offsetY = offsetArray[dimY * dimX + cy * dimX + cx];

	// Check if the thread is inside the frame (without offsets)
	if (cz < 3 && cy < dimY && cx < dimX) {
		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			warpedFrame[cz * dimY * dimX + newCy * dimX + newCx] = frame1[cz * dimY * dimX + cy * dimX + cx];
			atomicAdd(&hitCount[cz * dimY * dimX + cy * dimX + cx + (offsetY * dimX + offsetX)], ones[cz * dimY * dimX + cy * dimX + cx]);
		}
	}
}

// Kernel that removes artifacts from the warped frame
__global__ void artifactRemovalKernel(const unsigned char* frame1, const int* hitCount, unsigned char* warpedFrame, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Check if the thread is inside the frame (without offsets)
	if (cz < 3 && cy < dimY && cx < dimX) {
		// Check if the current pixel is inside the frame
		if (hitCount[cz * dimY * dimX + cy * dimX + cx] != 1) {
			warpedFrame[cz * dimY * dimX + cy * dimX + cx] = frame1[cz * dimY * dimX + cy * dimX + cx];
		}
	}
}

// Kernel that rearranges the image data from RGBRGBRGB... to each channel in a separate layer
__global__ void rearrangeImageDataRGBtoLayerOFC(const unsigned char* RGBArray, unsigned char* layerArray, const unsigned int dimZ, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int czi = cz;

	// Account for BGR to RGB
	if (cz == 0) {
		czi = 2;
	} else if (cz == 2) {
		czi = 0;
	}

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		layerArray[czi * dimY * dimX + cy * dimX + cx] = RGBArray[cz + (3 * cy * dimX) + (3 * cx)];
	}
}

// Kernel that rearranges the image data from each channel in a separate layer to RGBRGBRGB...
__global__ void rearrangeImageDataLayertoRGBOFC(const unsigned char* layerArray, unsigned char* RGBArray, const unsigned int dimZ, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;
	int czi = cz;

	// Account for BGR to RGB
	if (cz == 0) {
		czi = 2;
	} else if (cz == 2) {
		czi = 0;
	}

	// Check if result is within matrix boundaries
	if (cz < dimZ && cy < dimY && cx < dimX) {
		RGBArray[czi + (3 * cy * dimX) + (3 * cx)] = layerArray[cz * dimY * dimX + cy * dimX + cx];
	}
}

// Kernel that blends frame1 to frame2
__global__ void blendFrameKernel(const unsigned char* frame1, const unsigned char* frame2, unsigned char* blendedFrame, const double frame1Scalar, const double frame2Scalar, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;

	// Check if result is within matrix boundaries
	if (cz < 3 && cy < dimY && cx < dimX) {
		blendedFrame[cz * dimY * dimX + cy * dimX + cx] = static_cast<unsigned char>(static_cast<double>(frame1[cz * dimY * dimX + cy * dimX + cx]) * frame1Scalar + static_cast<double>(frame2[cz * dimY * dimX + cy * dimX + cx]) * frame2Scalar);
	}
}

// Constructor
OpticalFlowCalc::OpticalFlowCalc() = default;

/*
* Initializes the optical flow calculation
*
* @param dimY: The height of the frame
* @param dimX: The width of the frame
*/
void OpticalFlowCalc::init(const unsigned int dimY, const unsigned int dimX) {
	NUM_BLOCKS_X = fmaxf(ceilf(dimX / static_cast<float>(NUM_THREADS)), 1);
	NUM_BLOCKS_Y = fmaxf(ceilf(dimY / static_cast<float>(NUM_THREADS)), 1);
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
	imageDeltaArray.init({ 3, dimY, dimX });
	offsetArray.init({ 2, dimY, dimX });
	statusArray.init({ dimY, dimX });
	summedUpDeltaArray.init({ dimY, dimX });
	normalizedDeltaArrayA.init({ dimY, dimX });
	normalizedDeltaArrayB.init({ dimY, dimX });
	isValueDecreasedArray.init({ dimY, dimX });
	warpedFrame.init({ 3, dimY, dimX });
	hitCount.init({ 3, dimY, dimX });
	ones.init({ 3, dimY, dimX }, 1);
	layerFrame.init({ 3, dimY, dimX });
	RGBFrame.init({ 3, dimY, dimX });
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
*/
void OpticalFlowCalc::calculateOpticalFlow(const GPUArray<unsigned char>& frame1, const GPUArray<unsigned char>& frame2) {
	// Reset the arrays
	offsetArray.fill(0);
	summedUpDeltaArray.fill(0);

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (int iter = 0; iter < numIterations; iter++) {
		// Each step we adjust the offset array to find the ideal offset
		for (int step = 0; step < NUM_STEPS; step++) {
			// Calculate the image deltas with the current offset array
			calcImageDelta << <grid, threads3 >> > (frame1.arrayPtrGPU, frame2.arrayPtrGPU, imageDeltaArray.arrayPtrGPU, offsetArray.arrayPtrGPU, frame1.dimY, frame1.dimX);

			// Sum up the deltas of each window
			calcDeltaSums << <grid, threads3 >> > (imageDeltaArray.arrayPtrGPU, summedUpDeltaArray.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);

			// Switch between the two normalized delta arrays to avoid copying
			if (step % 2 == 0) {
				// Normalize the summed up delta array
				normalizeDeltaSums << <grid, threads1 >> > (summedUpDeltaArray.arrayPtrGPU, normalizedDeltaArrayB.arrayPtrGPU, offsetArray.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);

				// Check if the new normalized delta array is better than the old one
				compareArrays << <grid, threads1 >> > (normalizedDeltaArrayA.arrayPtrGPU, normalizedDeltaArrayB.arrayPtrGPU, isValueDecreasedArray.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);
			} else {
				// Normalize the summed up delta array
				normalizeDeltaSums << <grid, threads1 >> > (summedUpDeltaArray.arrayPtrGPU, normalizedDeltaArrayA.arrayPtrGPU, offsetArray.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);

				// Check if the new normalized delta array is better than the old one
				compareArrays << <grid, threads1 >> > (normalizedDeltaArrayB.arrayPtrGPU, normalizedDeltaArrayA.arrayPtrGPU, isValueDecreasedArray.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);
			}

			// Adjust the offset array based on the comparison results
			compositeOffsetArray << <grid, threads1 >> > (offsetArray.arrayPtrGPU, isValueDecreasedArray.arrayPtrGPU, statusArray.arrayPtrGPU, currentGlobalOffset, windowDimY, windowDimX, frame1.dimY, frame1.dimX);

			// Wait for all threads to finish
			cudaDeviceSynchronize();

			// Reset the summed up delta array
			summedUpDeltaArray.fill(0);
		}
		// Adjust window size
		windowDimX = fmax(windowDimX / 2, 1);
		windowDimY = fmax(windowDimY / 2, 1);

		// Adjust global offset
		currentGlobalOffset = fmax(currentGlobalOffset / 2, 1);

		// Reset the status array
		statusArray.fill(0);
	}

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Warps frame1 according to the offset array
*
* @param frame1: The frame to warp
* @param offsetArray: The array containing the offsets
*/
void OpticalFlowCalc::warpFrame(const GPUArray<unsigned char>& frame1) {
	// Reset the hit count array
	hitCount.fill(0);

	// Warp the frame
	rearrangeImageDataRGBtoLayerOFC << <grid, threads3 >> > (frame1.arrayPtrGPU, layerFrame.arrayPtrGPU, 3, frame1.dimY, frame1.dimX);
	warpFrameKernel << <grid, threads3 >> > (layerFrame.arrayPtrGPU, offsetArray.arrayPtrGPU, hitCount.arrayPtrGPU, ones.arrayPtrGPU, warpedFrame.arrayPtrGPU, frame1.dimY, frame1.dimX);
	rearrangeImageDataLayertoRGBOFC << <grid, threads3 >> > (warpedFrame.arrayPtrGPU, RGBFrame.arrayPtrGPU, 3, frame1.dimY, frame1.dimX);

	// Wait for all threads to finish
	cudaDeviceSynchronize();

	// Remove artifacts
	//artifactRemovalKernel << <grid, threads3 >> > (frame1.arrayPtrGPU, hitCount.arrayPtrGPU, warpedFrame.arrayPtrGPU, frame1.dimY, frame1.dimX);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Blends frame1 to frame2
*
* @param frame1: The frame to blend from
* @param frame2: The frame to blend to
* @param iIntFrameNum: The current interpolated frame number
* @param iNumSamples: The number of frames between frame1 and frame2
*/
void OpticalFlowCalc::blendFrames(const GPUArray<unsigned char>& frame1, const GPUArray<unsigned char>& frame2, int iIntFrameNum, int iNumSamples) {
	// Calculate the blend scalar
	const double frame2Scalar = static_cast<double>(iIntFrameNum) / static_cast<double>(iNumSamples);
	const double frame1Scalar = 1.0 - frame2Scalar;

	// Blend the frame
	blendFrameKernel << <grid, threads3 >> > (frame1.arrayPtrGPU, frame2.arrayPtrGPU, warpedFrame.arrayPtrGPU, frame1Scalar, frame2Scalar, frame1.dimY, frame1.dimX);

	// Wait for all threads to finish
	cudaDeviceSynchronize();

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}