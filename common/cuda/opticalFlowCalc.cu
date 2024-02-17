// Project Includes
#include "opticalFlowCalc.cuh"

// C++ libraries
#include <amvideo.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <string>

// Debug message function
void CudaDebugMessage(const std::string& message) {
	const std::string m_debugMessage = message + "\n";
	OutputDebugStringA(m_debugMessage.c_str());
}

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
			imageDeltaArray[cz + (3 * cy * dimX) + (3 * cx)] = 0;
		// Current pixel is inside of frame
		} else {
			imageDeltaArray[cz + (3 * cy * dimX) + (3 * cx)] = fabsf(frame1[cz + (3 * cy * dimX) + (3 * cx) + (3 * offsetY * dimX + 3 * offsetX)] - frame2[cz + (3 * cy * dimX) + (3 * cx)]);
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
		atomicAdd(&summedUpDeltaArray[(windowIndexY * windowDimY) * dimX + (windowIndexX * windowDimX)], imageDeltaArray[cz + (3 * cy * dimX) + (3 * cx)]);
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
__global__ void warpFrameKernel(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones, unsigned char* warpedFrame, const double frameScalar, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Get the current offsets to use
	const int offsetX = static_cast<int>(static_cast<double>(offsetArray[cy * dimX + cx]) * frameScalar);
	const int offsetY = static_cast<int>(static_cast<double>(offsetArray[dimY * dimX + cy * dimX + cx]) * frameScalar);

	// Check if the thread is inside the frame (without offsets)
	if (cz < 3 && cy < dimY && cx < dimX) {
		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			warpedFrame[cz + (3 * newCy * dimX) + (3 * newCx)] = frame1[cz + (3 * cy * dimX) + (3 * cx)];
			atomicAdd(&hitCount[cz + (3 * cy * dimX) + (3 * cx) + (3 * offsetY * dimX + 3 * offsetX)], ones[cz + (3 * cy * dimX) + (3 * cx)]);
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
		if (hitCount[cz + (3 * cy * dimX) + (3 * cx)] != 1) {
			warpedFrame[cz + (3 * cy * dimX) + (3 * cx)] = frame1[cz + (3 * cy * dimX) + (3 * cx)];
		}
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
		blendedFrame[cz + (3 * cy * dimX) + (3 * cx)] = static_cast<unsigned char>(static_cast<double>(frame1[cz + (3 * cy * dimX) + (3 * cx)]) * frame1Scalar + static_cast<double>(frame2[cz + (3 * cy * dimX) + (3 * cx)]) * frame2Scalar);
	}
}

// Kernel that blends frame1 to frame2
__global__ void testFrameKernel(const unsigned char* frame1, const unsigned char* frame2, unsigned char* blendedFrame, const double frame1Scalar, const double frame2Scalar, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;

	// Check if result is within matrix boundaries
	if (cz < 3 && cy < dimY && cx < dimX) {
		blendedFrame[cz + (3 * cy * dimX) + (3 * cx)] = static_cast<unsigned char>(static_cast<double>(frame1[cz + (3 * cy * dimX) + (3 * cx)]) * frame1Scalar);
	}
}

// Kernel that creates an HSV flow image from the offset array
__global__ void convertFlowToHSVKernel(const int* flowArray, unsigned char* RGBArray, const unsigned int dimZ, const unsigned int dimY, const unsigned int dimX, const double saturation, const double value, const float threshold) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;

	// Get the current flow values
	double x = flowArray[cy * dimX + cx];
	double y = -flowArray[dimY * dimX + cy * dimX + cx];

	// Check if the flow is below the threshold
	if (fabsf(x) < threshold && fabsf(y) < threshold) {
		RGBArray[cz + (3 * cy * dimX) + (3 * cx)] = 0;
		return;
	}

	x += 1;
	y += 1;

	// RGB struct
	struct RGB {
		int r, g, b;
	};

	// Calculate the angle in radians
	const double angle_rad = std::atan2(y, x);

	// Convert radians to degrees
	double angle_deg = angle_rad * (180.0 / 3.14159265359);

	// Ensure the angle is positive
	if (angle_deg < 0) {
		angle_deg += 360.0;
	}

	// Normalize the angle to the range [0, 360]
	angle_deg = fmod(angle_deg, 360.0);
	if (angle_deg < 0) {
		angle_deg += 360.0;
	}

	// Map the angle to the hue value in the HSV model
	const double hue = angle_deg / 360.0;

	// Convert HSV to RGB
	const int h_i = static_cast<int>(hue * 6);
	const double f = hue * 6 - h_i;
	const double p = value * (1 - saturation);
	const double q = value * (1 - f * saturation);
	const double t = value * (1 - (1 - f) * saturation);

	RGB rgb;
	switch (h_i % 6) {
		case 0: rgb = { static_cast<int>(value * 255), static_cast<int>(t * 255), static_cast<int>(p * 255) }; break;
		case 1: rgb = { static_cast<int>(q * 255), static_cast<int>(value * 255), static_cast<int>(p * 255) }; break;
		case 2: rgb = { static_cast<int>(p * 255), static_cast<int>(value * 255), static_cast<int>(t * 255) }; break;
		case 3: rgb = { static_cast<int>(p * 255), static_cast<int>(q * 255), static_cast<int>(value * 255) }; break;
		case 4: rgb = { static_cast<int>(t * 255), static_cast<int>(p * 255), static_cast<int>(value * 255) }; break;
		case 5: rgb = { static_cast<int>(value * 255), static_cast<int>(p * 255), static_cast<int>(q * 255) }; break;
	}

	// Write the RGB values to the array
	if (cz < dimZ && cy < dimY && cx < dimX) {
		// Blue
		if (cz == 0) {
			RGBArray[cz + (3 * cy * dimX) + (3 * cx)] = fminf(rgb.b, 255);
		// Green
		} else if (cz == 1) {
			RGBArray[cz + (3 * cy * dimX) + (3 * cx)] = fminf(rgb.g, 255);
		// Red
		} else {
			RGBArray[cz + (3 * cy * dimX) + (3 * cx)] = fminf(rgb.r, 255);
		}
	}
}

// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__global__ void flipFlowKernel(const int* flowArray12, int* flowArray21, const int dimY, const int dimX) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Get the current flow values
	const int x = flowArray12[cy * dimX + cx];
	const int y = flowArray12[dimY * dimX + cy * dimX + cx];

	// Project the flow values onto the flow array from frame 2 to frame 1
	if (cy < dimY && cx < dimX) {
		if (cz == 0 && (cy + y) < dimY && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
			flowArray21[(cy + y) * dimX + cx + x] = -x;
		} else if (cz == 1 && (cy + y) < dimY && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
			flowArray21[dimY * dimX + (cy + y) * dimX + cx + x] = -y;
		}
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
	frame1.init({ 3, dimY, dimX });
	frame2.init({ 3, dimY, dimX });
	imageDeltaArray.init({ 3, dimY, dimX });
	offsetArray12.init({ 2, dimY, dimX });
	offsetArray21.init({ 2, dimY, dimX });
	blurredOffsetArray12.init({ 2, dimY, dimX });
	blurredOffsetArray21.init({ 2, dimY, dimX });
	rgboffsetArray.init({ 3, dimY, dimX });
	statusArray.init({ dimY, dimX });
	summedUpDeltaArray.init({ dimY, dimX });
	normalizedDeltaArrayA.init({ dimY, dimX });
	normalizedDeltaArrayB.init({ dimY, dimX });
	isValueDecreasedArray.init({ dimY, dimX });
	warpedFrame12.init({ 3, dimY, dimX });
	warpedFrame21.init({ 3, dimY, dimX });
	blendedFrame.init({ 3, dimY, dimX });
	hitCount.init({ 3, dimY, dimX });
	ones.init({ 3, dimY, dimX }, 1);
	windowDimX = dimX;
	windowDimY = dimY;
	currentGlobalOffset = fmax(dimX / 192, 1);
	bIsInitialized = true;
}

/*
* Returns whether the optical flow calculation is initialized
*
* @return: True if the optical flow calculation is initialized, false otherwise
*/
bool OpticalFlowCalc::isInitialized() const {
	return bIsInitialized;
	
}

/*
* Updates the frame1 array
*/
void OpticalFlowCalc::updateFrame1(const BYTE* pInBuffer) {
	frame1.fillData(pInBuffer);
	bBisNewest = false;
}

/*
* Updates the frame2 array
*/
void OpticalFlowCalc::updateFrame2(const BYTE* pInBuffer) {
	frame2.fillData(pInBuffer);
	bBisNewest = true;
}

/*
* Calculates the optical flow between frame1 and frame2
*
* @param iNumIterations: Number of iterations to calculate the optical flow
* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
* @param iMaxOffsetDivider: The divider used to calculate the initial global offset
*
* @return: The time it took to calculate the optical flow
*/
double OpticalFlowCalc::calculateOpticalFlow(int iNumIterations, int iNumSteps, int iMaxOffsetDivider) {
	const auto start = std::chrono::high_resolution_clock::now();

	// Reset variables
	windowDimX = frame1.dimX;
	windowDimY = frame1.dimY;
	currentGlobalOffset = fmax(frame1.dimX / iMaxOffsetDivider, 1);
	if (iNumIterations == 0 || iNumIterations > ceil(log2f(frame1.dimX))) {
		iNumIterations = ceil(log2f(frame1.dimX));
	}

	// Reset the arrays
	offsetArray12.fill(0);
	summedUpDeltaArray.fill(0);

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		// Each step we adjust the offset array to find the ideal offset
		for (int step = 0; step < iNumSteps; step++) {
			// Calculate the image deltas with the current offset array
			if (bBisNewest) {
				calcImageDelta << <grid, threads3 >> > (frame1.arrayPtrGPU, frame2.arrayPtrGPU, imageDeltaArray.arrayPtrGPU, offsetArray12.arrayPtrGPU, frame1.dimY, frame1.dimX);
			} else {
				calcImageDelta << <grid, threads3 >> > (frame2.arrayPtrGPU, frame1.arrayPtrGPU, imageDeltaArray.arrayPtrGPU, offsetArray12.arrayPtrGPU, frame1.dimY, frame1.dimX);
			}
			// Sum up the deltas of each window
			calcDeltaSums << <grid, threads3 >> > (imageDeltaArray.arrayPtrGPU, summedUpDeltaArray.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);

			// Switch between the two normalized delta arrays to avoid copying
			if (step % 2 == 0) {
				// Normalize the summed up delta array
				normalizeDeltaSums << <grid, threads1 >> > (summedUpDeltaArray.arrayPtrGPU, normalizedDeltaArrayB.arrayPtrGPU, offsetArray12.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);

				// Check if the new normalized delta array is better than the old one
				compareArrays << <grid, threads1 >> > (normalizedDeltaArrayA.arrayPtrGPU, normalizedDeltaArrayB.arrayPtrGPU, isValueDecreasedArray.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);
			}
			else {
				// Normalize the summed up delta array
				normalizeDeltaSums << <grid, threads1 >> > (summedUpDeltaArray.arrayPtrGPU, normalizedDeltaArrayA.arrayPtrGPU, offsetArray12.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);

				// Check if the new normalized delta array is better than the old one
				compareArrays << <grid, threads1 >> > (normalizedDeltaArrayB.arrayPtrGPU, normalizedDeltaArrayA.arrayPtrGPU, isValueDecreasedArray.arrayPtrGPU, windowDimY, windowDimX, frame1.dimY, frame1.dimX);
			}

			// Adjust the offset array based on the comparison results
			compositeOffsetArray << <grid, threads1 >> > (offsetArray12.arrayPtrGPU, isValueDecreasedArray.arrayPtrGPU, statusArray.arrayPtrGPU, currentGlobalOffset, windowDimY, windowDimX, frame1.dimY, frame1.dimX);

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

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration<double, std::milli>(stop - start).count();
	return duration;
}

/*
* Warps frame1 according to the offset array to frame2
*
* @param offsetArray: The array containing the offsets
*/
void OpticalFlowCalc::warpFrame12(double dScalar) {
	const auto start = std::chrono::high_resolution_clock::now();

	// Calculate the blend scalar
	const double frameScalar = dScalar;

	// Reset the hit count array
	hitCount.fill(0);
	//offsetArray.fill(-offsetX, 0, frame1.dimY * frame1.dimX - 1);
	//offsetArray.fill(offsetY, frame1.dimY * frame1.dimX, 2 * frame1.dimY * frame1.dimX - 1);

	// Warp the frame
	if (bBisNewest) {
		warpFrameKernel << <grid, threads3 >> > (frame1.arrayPtrGPU, blurredOffsetArray12.arrayPtrGPU, hitCount.arrayPtrGPU, ones.arrayPtrGPU, warpedFrame12.arrayPtrGPU, frameScalar, frame1.dimY, frame1.dimX);
		//artifactRemovalKernel << <grid, threads3 >> > (frame1.arrayPtrGPU, hitCount.arrayPtrGPU, warpedFrame.arrayPtrGPU, frame1.dimY, frame1.dimX);
	} else {
		warpFrameKernel << <grid, threads3 >> > (frame2.arrayPtrGPU, blurredOffsetArray12.arrayPtrGPU, hitCount.arrayPtrGPU, ones.arrayPtrGPU, warpedFrame12.arrayPtrGPU, frameScalar, frame1.dimY, frame1.dimX);
		//artifactRemovalKernel << <grid, threads3 >> > (frame2.arrayPtrGPU, hitCount.arrayPtrGPU, warpedFrame.arrayPtrGPU, frame1.dimY, frame1.dimX);
	}

	// Wait for all threads to finish
	cudaDeviceSynchronize();

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration<double, std::milli>(stop - start).count();
	//CudaDebugMessage("\nWarp Time: " + std::to_string(duration) + " milliseconds");
}

/*
* Warps frame2 according to the offset array to frame1
*
* @param offsetArray: The array containing the offsets
*/
void OpticalFlowCalc::warpFrame21(double dScalar) {
	const auto start = std::chrono::high_resolution_clock::now();

	// Calculate the blend scalar
	const double frameScalar = 1.0 - dScalar;

	// Reset the hit count array
	hitCount.fill(0);
	//offsetArray.fill(-offsetX, 0, frame1.dimY * frame1.dimX - 1);
	//offsetArray.fill(offsetY, frame1.dimY * frame1.dimX, 2 * frame1.dimY * frame1.dimX - 1);

	// Warp the frame
	if (bBisNewest) {
		warpFrameKernel << <grid, threads3 >> > (frame2.arrayPtrGPU, blurredOffsetArray21.arrayPtrGPU, hitCount.arrayPtrGPU, ones.arrayPtrGPU, warpedFrame21.arrayPtrGPU, frameScalar, frame1.dimY, frame1.dimX);
		//artifactRemovalKernel << <grid, threads3 >> > (frame1.arrayPtrGPU, hitCount.arrayPtrGPU, warpedFrame.arrayPtrGPU, frame1.dimY, frame1.dimX);
	} else {
		warpFrameKernel << <grid, threads3 >> > (frame1.arrayPtrGPU, blurredOffsetArray21.arrayPtrGPU, hitCount.arrayPtrGPU, ones.arrayPtrGPU, warpedFrame21.arrayPtrGPU, frameScalar, frame1.dimY, frame1.dimX);
		//artifactRemovalKernel << <grid, threads3 >> > (frame2.arrayPtrGPU, hitCount.arrayPtrGPU, warpedFrame.arrayPtrGPU, frame1.dimY, frame1.dimX);
	}

	// Wait for all threads to finish
	cudaDeviceSynchronize();

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration<double, std::milli>(stop - start).count();
	//CudaDebugMessage("\nWarp Time: " + std::to_string(duration) + " milliseconds");
}

/*
* Blends warpedFrame1 to warpedFrame2
*
* @param dScalar: The scalar to blend the frames with
*/
void OpticalFlowCalc::blendFrames(double dScalar) {
	// Calculate the blend scalar
	const double frame1Scalar = 1.0 - dScalar;
	const double frame2Scalar = dScalar;

	// Blend the frames
	blendFrameKernel << <grid, threads3 >> > (warpedFrame12.arrayPtrGPU, warpedFrame21.arrayPtrGPU, blendedFrame.arrayPtrGPU, frame1Scalar, frame2Scalar, warpedFrame12.dimY, warpedFrame12.dimX);

	// Wait for all threads to finish
	cudaDeviceSynchronize();

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Downloads the array as a flow image into the output memory pointer
*
* @param memPointer: Pointer to the memory to transfer the array to
* @param saturation: The saturation of the flow image
* @param value: The value of the flow image
* @param threshold: The threshold to use for the flow image
*/
void OpticalFlowCalc::downloadFlowAsHSV(unsigned char* memPointer, const double saturation, const double value, const float threshold) const {
	// Launch kernel
	convertFlowToHSVKernel << <grid, threads3 >> > (offsetArray12.arrayPtrGPU, rgboffsetArray.arrayPtrGPU, 3, offsetArray12.dimY, offsetArray12.dimX, saturation, value, threshold);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Copy host array to GPU
	cudaMemcpy(memPointer, rgboffsetArray.arrayPtrGPU, 3 * offsetArray12.dimY * offsetArray12.dimX, cudaMemcpyDeviceToHost);
}

/*
* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
*
* @param memPointer: Pointer to the memory to transfer the array to
* @param saturation: The saturation of the flow image
* @param value: The value of the flow image
* @param threshold: The threshold to use for the flow image
*/
void OpticalFlowCalc::flipFlow() {
	// Calculate the number of blocks needed
	const int NUM_BLOCKS_X = fmaxf(ceilf(offsetArray12.dimX / static_cast<float>(NUM_THREADS)), 1);
	const int NUM_BLOCKS_Y = fmaxf(ceilf(offsetArray12.dimY / static_cast<float>(NUM_THREADS)), 1);
	const int NUM_BLOCKS_Z = 1;

	// Use dim3 structs for block and grid size
	dim3 grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
	dim3 threads(NUM_THREADS, NUM_THREADS, 2);

	// Reset the offset array
	offsetArray21.fill(0);

	// Launch kernel
	flipFlowKernel << <grid, threads >> > (offsetArray12.arrayPtrGPU, offsetArray21.arrayPtrGPU, offsetArray12.dimY, offsetArray12.dimX);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}