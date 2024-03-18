#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include "opticalFlowCalc.cuh"

// Kernel that blurs a frame
template <typename T>
__global__ void blurFrameKernel(const T* frameArray, T* blurredFrameArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char lumStart,
								const unsigned char lumEnd, const unsigned short lumPixelCount, const char chromStart, 
								const unsigned char chromEnd, const unsigned short chromPixelCount, const unsigned short dimY, const unsigned short dimX) {
	// Shared memory for the frame to prevent multiple global memory accesses
	extern __shared__ unsigned char sharedFrameArray[];

	// Current entry to be computed by the thread
	const unsigned short cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned short cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned char cz = blockIdx.z;

	// Check if the current thread is supposed to perform calculations
	if (cz == 1 && (cy >= dimY / 2 || cx >= dimX)) {
		return;
	}

	const unsigned short trX = blockIdx.x * blockDim.x;
	const unsigned short trY = blockIdx.y * blockDim.y;
	unsigned char offsetX;
	unsigned char offsetY;

    // Calculate the number of entries to fill for this thread
    const unsigned short threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned char entriesToFill = avgEntriesPerThread + (threadIndex < remainder ? 1 : 0); // 4 (first 6 threads) / 3 (remaining threads) 

    // Calculate the starting index for this thread
    unsigned short startIndex = 0;
    for (unsigned short i = 0; i < threadIndex; ++i) {
        startIndex += avgEntriesPerThread + (i < remainder ? 1 : 0);
    }

    // Fill the shared memory for this thread
    for (unsigned short i = 0; i < entriesToFill; ++i) {
		offsetX = (startIndex + i) % chacheSize;
		offsetY = (startIndex + i) / chacheSize;
		if ((trY - boundsOffset + offsetY) < dimY && (trX - boundsOffset + offsetX) < dimX) {
			sharedFrameArray[startIndex + i] = frameArray[cz * dimY * dimX + (trY - boundsOffset + offsetY) * dimX + (trX - boundsOffset + offsetX)];
		} else {
			sharedFrameArray[startIndex + i] = 0;
		}
	}

    // Ensure all threads have finished loading before continuing
    __syncthreads();

	// Calculate the x and y boundaries of the kernel
	unsigned int blurredPixel = 0;

	// Collect the sum of the surrounding pixels
	// Y-Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		for (char y = lumStart; y < lumEnd; y++) {
			for (char x = lumStart; x < lumEnd; x++) {
				if ((cy + y) < dimY && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
					blurredPixel += sharedFrameArray[(threadIdx.y + boundsOffset + y) * chacheSize + threadIdx.x + boundsOffset + x];
				} else {
					blurredPixel += sharedFrameArray[(threadIdx.y + boundsOffset) * chacheSize + threadIdx.x + boundsOffset];
				}
			}
		}
		blurredPixel /= lumPixelCount;
		blurredFrameArray[cz * dimY * dimX + cy * dimX + cx] = blurredPixel;
	// U/V-Channel
	} else if (cz == 1 && cy < dimY / 2 && cx < dimX) {
		for (char y = chromStart; y < chromEnd; y++) {
			for (char x = chromStart; x < chromEnd; x++) {
				if ((cy + y) < dimY / 2 && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
					blurredPixel += sharedFrameArray[(threadIdx.y + boundsOffset + y) * chacheSize + threadIdx.x + boundsOffset + x * 2];
				} else {
					blurredPixel += sharedFrameArray[(threadIdx.y + boundsOffset) * chacheSize + threadIdx.x + boundsOffset];
				}
			}
		}
		blurredPixel /= chromPixelCount;
		blurredFrameArray[cz * dimY * dimX + cy * dimX + cx] = blurredPixel;
	}
}

// Kernel that sets the initial offset array
__global__ void setInitialOffset(int* offsetArray, const unsigned int dimZ, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	if (cz < dimZ && cy < dimY && cx < dimX) {
		// Set the Y direction to no offset
		offsetArray[dimZ * dimY * dimX + cz * dimY * dimX + cy * dimX + cx] = 0;

		// Set the X direction layer 0 to no offset
		if (cz == 0) {
			offsetArray[cy * dimX + cx] = 0;
		// Set the X direction layer 1 to a -2 offset
		} else if (cz == 1) {
			offsetArray[dimY * dimX + cy * dimX + cx] = -2;
		// Set the X direction layer 2 to a -1 offset
		} else if (cz == 2) {
			offsetArray[2 * dimY * dimX + cy * dimX + cx] = -1;
		// Set the X direction layer 3 to a +1 offset
		} else if (cz == 3) {
			offsetArray[3 * dimY * dimX + cy * dimX + cx] = 1;
		// Set the X direction layer 4 to a +2 offset
		} else if (cz == 4) {
			offsetArray[4 * dimY * dimX + cy * dimX + cx] = 2;
		}
	}
}

// Kernel that calculates the absolute difference between two frames using the offset array
template <typename T>
__global__ void calcImageDelta(const T* frame1, const T* frame2, T* imageDeltaArray,
							   const int* offsetArray, const unsigned int lowDimY, const unsigned int lowDimX, 
							   const unsigned int dimY, const unsigned int dimX, const float resolutionScalar, const unsigned int directionIdxOffset, 
							   const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;

	// Is the current thread supposed to perform calculations
	if (cy < lowDimY && cx < lowDimX) {
		const unsigned int layerOffset = blockIdx.z * lowDimY * lowDimX; // Offset to index the layer of the current thread
		const unsigned int scaledCx = static_cast<unsigned int>(static_cast<float>(cx) * resolutionScalar); // The X-Index of the current thread in the input frames
		const unsigned int scaledCy = static_cast<unsigned int>(static_cast<float>(cy) * resolutionScalar); // The Y-Index of the current thread in the input frames
		const unsigned int evenCx = cx & ~1; // The X-Index of the current thread rounded to be even

		const unsigned int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim
		const unsigned int threadIndex3D = cz * lowDimY * lowDimX + threadIndex2D; // Standard thread index

		// Y-Channel
		if (threadIdx.z == 0) {
			const int offsetX = -offsetArray[layerOffset + threadIndex2D];
			const int offsetY = -offsetArray[directionIdxOffset + layerOffset + threadIndex2D];
			const int newCx = scaledCx + offsetX;
			const int newCy = scaledCy + offsetY;

			imageDeltaArray[threadIndex3D] = (newCy < 0 || newCx < 0 || newCy >= dimY || newCx >= dimX) ? 0 : 
				abs(frame1[newCy * dimX + newCx] - frame2[scaledCy * dimX + scaledCx]);


		// U/V-Channel
		} else {
			const int offsetX = -offsetArray[layerOffset + cy * lowDimX + evenCx];
			const int offsetY = -offsetArray[directionIdxOffset + layerOffset + cy * lowDimX + evenCx];
			const int newCx = scaledCx + (offsetX & ~1);
			const int newCy = (scaledCy >> 1) + (offsetY >> 1);

			imageDeltaArray[threadIndex3D] = (newCy < 0 || newCx < 0 || newCy >= (dimY >> 1) || newCx >= dimX) ? 0 : 
				abs(frame1[channelIdxOffset + newCy * dimX + newCx] - frame2[channelIdxOffset + (scaledCy >> 1) * dimX + scaledCx]) << 1;
		}
	}
}

// Helper kernel for the calcDeltaSums kernel
__device__ void warpReduce8x8(volatile unsigned int* partial_sums, int tIdx) {
	partial_sums[tIdx] += partial_sums[tIdx + 32];
	partial_sums[tIdx] += partial_sums[tIdx + 16];
	partial_sums[tIdx] += partial_sums[tIdx + 8];
	partial_sums[tIdx] += partial_sums[tIdx + 4];
	partial_sums[tIdx] += partial_sums[tIdx + 2];
	partial_sums[tIdx] += partial_sums[tIdx + 1];
}

// Helper kernel for the calcDeltaSums kernel
__device__ void warpReduce4x4(volatile unsigned int* partial_sums, int tIdx) {
	partial_sums[tIdx] += partial_sums[tIdx + 16];
	partial_sums[tIdx] += partial_sums[tIdx + 8];
	partial_sums[tIdx] += partial_sums[tIdx + 2];
	partial_sums[tIdx] += partial_sums[tIdx + 1];
}

// Helper kernel for the calcDeltaSums kernel
__device__ void warpReduce2x2(volatile unsigned int* partial_sums, int tIdx) {
	partial_sums[tIdx] += partial_sums[tIdx + 8];
	partial_sums[tIdx] += partial_sums[tIdx + 1];
}

// Kernel that sums up all the pixel deltas of each window for window sizes of at least 8x8
template <typename T>
__global__ void calcDeltaSums8x8(const T* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
						const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim) {
	// Handle used to synchronize all threads
	auto g = cooperative_groups::this_thread_block();

	// Shared memory for the partial sums of the current block
	extern __shared__ unsigned int partial_sums[];

	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z;
	const unsigned int tIdx = threadIdx.y * blockDim.x + threadIdx.x;
	partial_sums[tIdx] = 0;

	if (cy < lowDimY && cx < lowDimX) {
		// Add the luminace and color channel together
		partial_sums[tIdx] = imageDeltaArray[cz * layerIdxOffset + cy * lowDimX + cx] + imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + cy * lowDimX + cx];
		__syncthreads();

		// Sum up the remaining pixels for the current window
		for (int s = (blockDim.y * blockDim.x) >> 1; s > 32; s >>= 1) {
			if (tIdx < s) {
				partial_sums[tIdx] += partial_sums[tIdx + s];
			}
			__syncthreads();
		}

		// Loop over the remaining values
		if (tIdx < 32) {
			warpReduce8x8(partial_sums, tIdx);
		}

		// Sync all threads
		g.sync();

		// Sum up the results of all blocks
		if (tIdx == 0) {
			const unsigned int windowIndexX = cx / windowDim;
			const unsigned int windowIndexY = cy / windowDim;
			atomicAdd(&summedUpDeltaArray[cz * channelIdxOffset + (windowIndexY * windowDim) * lowDimX + (windowIndexX * windowDim)], partial_sums[0]);
		}
	}
}

// Kernel that sums up all the pixel deltas of each window for window sizes of 4x4
template <typename T>
__global__ void calcDeltaSums4x4(const T* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
						const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim) {
	// Handle used to synchronize all threads
	auto g = cooperative_groups::this_thread_block();

	// Shared memory for the partial sums of the current block
	extern __shared__ unsigned int partial_sums[];

	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;
	const unsigned int tIdx = threadIdx.y * blockDim.x + threadIdx.x;
	partial_sums[tIdx] = 0;

	if (cy < lowDimY && cx < lowDimX) {
		// Add the luminace and color channel together
		partial_sums[tIdx] = imageDeltaArray[cz * layerIdxOffset + cy * lowDimX + cx] + imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + cy * lowDimX + cx];
		__syncthreads();

		// Top 4x4 Blocks
		if (threadIdx.y < 2) {
			warpReduce4x4(partial_sums, tIdx);
		// Bottom 4x4 Blocks
		} else if (threadIdx.y >= 4 && threadIdx.y < 6) {
			warpReduce4x4(partial_sums, tIdx);
		}

		// Sync all threads
		g.sync();

		// Sum up the results of all blocks
		if (tIdx == 0 || tIdx == 4 || tIdx == 32 || tIdx == 36) {
			const unsigned int windowIndexX = cx / windowDim;
			const unsigned int windowIndexY = cy / windowDim;
			atomicAdd(&summedUpDeltaArray[cz * channelIdxOffset + (windowIndexY * windowDim) * lowDimX + (windowIndexX * windowDim)], partial_sums[tIdx]);
		}
	}
}

// Kernel that sums up all the pixel deltas of each window for window sizes of 2x2
template <typename T>
__global__ void calcDeltaSums2x2(const T* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
						const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim) {
	// Handle used to synchronize all threads
	auto g = cooperative_groups::this_thread_block();

	// Shared memory for the partial sums of the current block
	extern __shared__ unsigned int partial_sums[];

	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;
	const unsigned int tIdx = threadIdx.y * blockDim.x + threadIdx.x;
	partial_sums[tIdx] = 0;

	if (cy < lowDimY && cx < lowDimX) {
		// Add the luminace and color channel together
		partial_sums[tIdx] = imageDeltaArray[cz * layerIdxOffset + cy * lowDimX + cx] + imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + cy * lowDimX + cx];
		__syncthreads();

		if ((threadIdx.y & 1) == 0) {
			warpReduce2x2(partial_sums, tIdx);
		}

		// Sync all threads
		g.sync();

		// Sum up the results of all blocks
		if ((tIdx & 1) == 0 && (threadIdx.y & 1) == 0) {
			const unsigned int windowIndexX = cx / windowDim;
			const unsigned int windowIndexY = cy / windowDim;
			atomicAdd(&summedUpDeltaArray[cz * channelIdxOffset + (windowIndexY * windowDim) * lowDimX + (windowIndexX * windowDim)], partial_sums[tIdx]);
		}
	}
}

// Kernel that sums up all the pixel deltas of each window for window sizes of 1x1
template <typename T>
__global__ void calcDeltaSums1x1(const T* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
						const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z * blockDim.z + threadIdx.z;

	if (cy < lowDimY && cx < lowDimX) {
		// Add the luminace and color channel together
		summedUpDeltaArray[cz * channelIdxOffset + cy * lowDimX + cx] = imageDeltaArray[cz * layerIdxOffset + cy * lowDimX + cx] + imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + cy * lowDimX + cx];
	}
}

// Kernel that normalizes all the pixel deltas of each window
__global__ void normalizeDeltaSums(const unsigned int* summedUpDeltaArray, unsigned char* globalLowestLayerArray,
                                   const int* offsetArray, const unsigned int windowDim,
								   const int dimZ, const int dimY, const int dimX) {
	// Allocate shared memory to share values across layers
	__shared__ double normalizedDeltaArray[5 * NUM_THREADS * NUM_THREADS * 8];
	
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;
	bool isWindowRepresent = false;

	// Check if the thread is a window represent
	if (cy % windowDim == 0 && cx % windowDim == 0) {
		isWindowRepresent = true;
		// Get the current window information
		const int offsetX = offsetArray[cz * dimY * dimX + cy * dimX + cx];
		const int offsetY = offsetArray[dimZ * dimY * dimX + cz * dimY * dimX + cy * dimX + cx];

		// Calculate the number of pixels in the window
		unsigned int numPixels = windowDim * windowDim;

		// Calculate the not overlapping pixels
		int overlapX;
		int overlapY;

		// Calculate the number of not overlapping pixels
		if ((cx + windowDim + fabsf(offsetX) > dimX) || (cx - offsetX < 0)) {
			overlapX = fabsf(offsetX);
		} else {
			overlapX = 0;
		}

		if ((cy + windowDim + fabsf(offsetY) > dimY) || (cy - offsetY < 0)) {
			overlapY = fabsf(offsetY);
		} else {
			overlapY = 0;
		}

		const int numNotOverlappingPixels = overlapY * overlapX;
		numPixels -= numNotOverlappingPixels;

		// Normalize the summed up delta
		normalizedDeltaArray[cz * NUM_THREADS * NUM_THREADS + threadIdx.y * NUM_THREADS + threadIdx.x] = static_cast<double>(summedUpDeltaArray[cz * dimY * dimX + cy * dimX + cx]) / static_cast<double>(numPixels);
	}

	// Wait for all threads to finish filling the values
	__syncthreads();

	// Find the layer with the lowest value
	if (cz == 0 && isWindowRepresent) {
		unsigned char lowestLayer = 0;

		for (unsigned char z = 1; z < dimZ; ++z) {
			if (normalizedDeltaArray[z * NUM_THREADS * NUM_THREADS + threadIdx.y * NUM_THREADS + threadIdx.x] < 
				normalizedDeltaArray[lowestLayer * NUM_THREADS * NUM_THREADS + threadIdx.y * NUM_THREADS + threadIdx.x]) {
				lowestLayer = z;
			}
		}

		globalLowestLayerArray[cy * dimX + cx] = lowestLayer;
	}
}

// Kernel that adjusts the offset array based on the comparison results
__global__ void adjustOffsetArray(int* offsetArray, const unsigned char* globalLowestLayerArray, unsigned char* statusArray,
								  const unsigned int windowDim, const unsigned int dimZ, 
								  const unsigned int dimY, const unsigned int dimX, const bool lastRun) {

	// Allocate shared memory to cache the lowest layer
	__shared__ unsigned char lowestLayerArray[NUM_THREADS * NUM_THREADS];

	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;

	/*
	* Status Array Key:
	* 0: Find the initial x direction
	* 1: Find extended positive x direction
	* 2: Find extended negative x direction
	* 3: Find the initial y direction
	* 4: Find extended positive y direction
	* 5: Find extended negative y direction
	* 6: Search complete
	*/

	if (cy < dimY && cx < dimX) {
		const unsigned int trwx = (((cx / blockDim.x) * blockDim.x) / windowDim) * windowDim;
		const unsigned int trwy = (((cy / blockDim.y) * blockDim.y) / windowDim) * windowDim;
		const unsigned int wx = (cx / windowDim) * windowDim;
		const unsigned int wy = (cy / windowDim) * windowDim;
		unsigned char lowestLayer;

		// We are the block representative
		if (threadIdx.y == 0 && threadIdx.x == 0) {
			lowestLayer = globalLowestLayerArray[wy * dimX + wx];
			lowestLayerArray[0] = lowestLayer;
		}
		
		__syncthreads();

		// We can reuse the block representative value
		if (wy == trwy && wx == trwx) {
			lowestLayer = lowestLayerArray[0];
		// The value relevant to us is different from the cached one
		} else {
			lowestLayer = globalLowestLayerArray[wy * dimX + wx];
		}

		const unsigned char currentStatus = statusArray[cy * dimX + cx];
		int currX;
		int currY;

		// If this is the last run, we need to adjust the offset array accordingly
		switch (currentStatus) {
			// We are still trying to find the perfect x direction
			case 0:
			case 1:
			case 2:
				currX = offsetArray[lowestLayer * dimY * dimX + cy * dimX + cx];

				// Shift the X direction layer 0 to the ideal X direction
				offsetArray[dimY * dimX + cy * dimX + cx] = currX;
				// Shift the X direction layer 1 by -2
				offsetArray[dimY * dimX + cy * dimX + cx] = currX - 2;
				// Shift the X direction layer 2 by -1
				offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX - 1;
				// Shift the X direction layer 3 by +1
				offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX + 1;
				// Shift the X direction layer 4 by +2
				offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX + 2;
				break;

			// We are still trying to find the perfect y direction
			case 3:
			case 4:
			case 5:
				currX = offsetArray[cy * dimX + cx];
				currY = offsetArray[dimZ * dimY * dimX + lowestLayer * dimY * dimX + cy * dimX + cx];

				// Set all Y direction layers to the ideal Y direction
				for (unsigned int z = 0; z < dimZ; z++) {
					offsetArray[dimZ * dimY * dimX + z * dimY * dimX + cy * dimX + cx] = currY;
				}

				// Shift the X direction layer 1 by -2
				offsetArray[dimY * dimX + cy * dimX + cx] = currX - 2;
				// Shift the X direction layer 2 by -1
				offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX - 1;
				// Shift the X direction layer 3 by +1
				offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX + 1;
				// Shift the X direction layer 4 by +2
				offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX + 2;
				break;

			// Search completed, so we adjust the offset array for the next run
			default:
				currX = offsetArray[cy * dimX + cx];
				currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];

				// Set all Y direction layers to the ideal Y direction
				for (unsigned int z = 1; z < dimZ; z++) {
					offsetArray[dimZ * dimY * dimX + z * dimY * dimX + cy * dimX + cx] = currY;
				}

				// Shift the X direction layer 1 by -2
				offsetArray[dimY * dimX + cy * dimX + cx] = currX - 2;
				// Shift the X direction layer 2 by -1
				offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX - 1;
				// Shift the X direction layer 3 by +1
				offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX + 1;
				// Shift the X direction layer 4 by +2
				offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX + 2;
				break;
		}

		// If we are still calculating, adjust the offset array based on the current status and lowest layer
		switch (currentStatus) {
			/*
			* X - DIRECTION
			*/
			// Find the initial x direction
			case 0:
				// If the lowest layer is 0, no x direction is needed -> continue to y direction
				if (lowestLayer == 0) {
					statusArray[cy * dimX + cx] = 3;
					currX = offsetArray[cy * dimX + cx];
					currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					for (int z = 0; z < dimZ; z++) {
						offsetArray[z * dimY * dimX + cy * dimX + cx] = currX;
					}
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 2, ideal x direction found -> continue to y direction
				} else if (lowestLayer == 2) {
					statusArray[cy * dimX + cx] = 3;
					currX = offsetArray[2 * dimY * dimX + cy * dimX + cx];
					currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[cy * dimX + cx] = currX;
					offsetArray[dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 3, ideal x direction found -> continue to y direction
				} else if (lowestLayer == 3) {
					statusArray[cy * dimX + cx] = 3;
					currX = offsetArray[3 * dimY * dimX + cy * dimX + cx];
					currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[cy * dimX + cx] = currX;
					offsetArray[dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 4 -> continue moving in the positive x direction
				} else if (lowestLayer == 4) {
					statusArray[cy * dimX + cx] = 1;
					currX = offsetArray[4 * dimY * dimX + cy * dimX + cx];
					offsetArray[cy * dimX + cx] = currX + 4;
					offsetArray[dimY * dimX + cy * dimX + cx] = currX + 3;
					offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX + 2;
					offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX + 1;

				// If the lowest layer is 1 -> continue moving in the negative x direction
				} else if (lowestLayer == 1) {
					statusArray[cy * dimX + cx] = 2;
					currX = offsetArray[dimY * dimX + cy * dimX + cx];
					offsetArray[cy * dimX + cx] = currX;
					offsetArray[dimY * dimX + cy * dimX + cx] = currX - 1;
					offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX - 2;
					offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX - 3;
					offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX - 4;
				}
				break;

			// Find extended positive x direction
			case 1:
				// If the lowest layer is not 0, no x further direction is needed -> continue to y direction
				if (lowestLayer != 0) {
					statusArray[cy * dimX + cx] = 3;
					const int idealX = offsetArray[lowestLayer * dimY * dimX + cy * dimX + cx];
					for (unsigned int z = 0; z < dimZ; z++) {
						offsetArray[z * dimY * dimX + cy * dimX + cx] = idealX;
					}
					currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 0 -> continue moving in x direction
				} else {
					currX = offsetArray[cy * dimX + cx];
					offsetArray[cy * dimX + cx] = currX + 4;
					offsetArray[dimY * dimX + cy * dimX + cx] = currX + 3;
					offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX + 2;
					offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX + 1;
					offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX;
				}
				break;

			// Find extended negative x direction
			case 2:
				// If the lowest layer is not 4, no x further direction is needed -> continue to y direction
				if (lowestLayer != 4) {
					statusArray[cy * dimX + cx] = 3;
					const int idealX = offsetArray[lowestLayer * dimY * dimX + cy * dimX + cx];
					for (unsigned int z = 0; z < dimZ; z++) {
						offsetArray[z * dimY * dimX + cy * dimX + cx] = idealX;
					}
					currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 4 -> continue moving in x direction
				} else {
					currX = offsetArray[4 * dimY * dimX + cy * dimX + cx];
					offsetArray[cy * dimX + cx] = currX;
					offsetArray[dimY * dimX + cy * dimX + cx] = currX - 1;
					offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX - 2;
					offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX - 3;
					offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX - 4;
				}
				break;

			/*
			* Y - DIRECTION
			*/
			// Find the initial y direction
			case 3:
				// If the lowest layer is 0, 2, or 3, no y direction is needed -> we are done
				if (lowestLayer == 0 || lowestLayer == 2 || lowestLayer == 3) {
					statusArray[cy * dimX + cx] = 6;
					if (lowestLayer != 0) {
						currY = offsetArray[dimZ * dimY * dimX + lowestLayer * dimY * dimX + cy * dimX + cx];
						offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY;
					}

				// If the lowest layer is 4 -> continue moving in the positive y direction
				} else if (lowestLayer == 4) {
					statusArray[cy * dimX + cx] = 4;
					currY = offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY + 4;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY + 3;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY + 2;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;

				// If the lowest layer is 1 -> continue moving in the negative y direction
				} else if (lowestLayer == 1) {
					statusArray[cy * dimX + cx] = 5;
					currY = offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY - 3;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY - 4;
				}
				break;

			// Find extended positive y direction
			case 4:
				// If the lowest layer is not 0, no y further direction is needed -> we are done
				if (lowestLayer != 0) {
					statusArray[cy * dimX + cx] = 6;
					const int idealY = offsetArray[dimZ * dimY * dimX + lowestLayer * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = idealY;

				// If the lowest layer is 0 -> continue moving in y direction
				} else {
					currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY + 4;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY + 3;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY + 2;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY;
				}
				break;

			// Find extended negative y direction
			case 5:
				// If the lowest layer is not 4, no y further direction is needed -> we are done
				if (lowestLayer != 4) {
					statusArray[cy * dimX + cx] = 6;
					const int idealY = offsetArray[dimZ * dimY * dimX + lowestLayer * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = idealY;

				// If the lowest layer is 4 -> continue moving in y direction
				} else {
					currY = offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY - 3;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY - 4;
				}
				break;

			// Search is complete
			default:
				break;
		}
	}
}

// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__global__ void flipFlowKernel(const int* flowArray12, int* flowArray21, const unsigned int dimZ,
							   const int dimY, const int dimX, const double dResolutionDivider) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Check if we are inside the flow array
	if (cz < 2 && cy < dimY && cx < dimX) {
		// Get the current flow values
		const int x = flowArray12[cy * dimX + cx];
		const int y = flowArray12[dimZ * dimY * dimX + cy * dimX + cx];
		const int scaledX = static_cast<int>(x * dResolutionDivider);
		const int scaledY = static_cast<int>(y * dResolutionDivider);

		// Project the flow values onto the flow array from frame 2 to frame 1
		// X-Layer
		if (cz == 0 && (cy + scaledY) < dimY && (cy + scaledY) >= 0 && (cx + scaledX) < dimX && (cx + scaledX) >= 0) {
			flowArray21[(cy + scaledY) * dimX + cx + scaledX] = -x;
		// Y-Layer
		} else if (cz == 1 && (cy + scaledY) < dimY && (cy + scaledY) >= 0 && (cx + scaledX) < dimX && (cx + scaledX) >= 0) {
			flowArray21[dimY * dimX + (cy + scaledY) * dimX + cx + scaledX] = -y;
		}
	}
}

// Kernel that blurs a flow array
__global__ void blurFlowKernel(const int* flowArray, int* blurredFlowArray, const int kernelSize, const int dimZ, const int dimY,
						   const int dimX, const bool offset12) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	if (kernelSize > 1) {
		// Calculate the x and y boundaries of the kernel
		const int start = -(kernelSize / 2);
		const int end = (kernelSize / 2);
		long blurredOffset = 0;

		// Collect the sum of the surrounding pixels
		if (cz < 2 && cy < dimY && cx < dimX) {
			for (int y = start; y < end; y++) {
				for (int x = start; x < end; x++) {
					if ((cy + y) < dimY && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
						blurredOffset += flowArray[cz * dimZ * dimY * dimX + (cy + y) * dimX + cx + x];
					}
				}
			}
			blurredOffset /= (end - start) * (end - start);
			blurredFlowArray[cz * dimY * dimX + cy * dimX + cx] = blurredOffset;
		}
	} else {
		if (cz < 2 && cy < dimY && cx < dimX) {
			if (offset12) {
				blurredFlowArray[cz * dimY * dimX + cy * dimX + cx] = flowArray[cz * dimZ * dimY * dimX + cy * dimX + cx];
			} else {
				blurredFlowArray[cz * dimY * dimX + cy * dimX + cx] = flowArray[cz * dimY * dimX + cy * dimX + cx];
			}
		}
	}
}

// Kernel that removes artifacts from the warped frame
template <typename T>
__global__ void artifactRemovalKernelForBlending(const T* frame1, const int* hitCount, T* warpedFrame,
												    const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		if (hitCount[cy * dimX + cx] != 1) {
			warpedFrame[cy * dimX + cx] = frame1[cy * dimX + cx];
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		if (hitCount[cy * dimX + cx] != 1) {
			warpedFrame[dimY * dimX + cy * dimX + cx] = frame1[dimY * dimX + cy * dimX + cx];
		}
	}
}

/*
* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
*/
void OpticalFlowCalc::flipFlow() const {
	// Reset the offset array
	m_offsetArray21.zero();

	// Launch kernel
	flipFlowKernel << <m_lowGrid, m_threads2 >> > (m_offsetArray12.arrayPtrGPU, m_offsetArray21.arrayPtrGPU,
												   m_iNumLayers, m_iLowDimY, m_iLowDimX, m_dResolutionDivider);
}

/*
* Blurs the offset arrays
*
* @param kernelSize: Size of the kernel to use for the blur
*/
void OpticalFlowCalc::blurFlowArrays(const int kernelSize) const {
	// Create CUDA streams
	cudaStream_t blurStream1, blurStream2;
	cudaStreamCreate(&blurStream1);
	cudaStreamCreate(&blurStream2);

	// Launch kernels
	blurFlowKernel << <m_lowGrid, m_threads2, 0, blurStream1 >> > (m_offsetArray12.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU, kernelSize, m_iNumLayers, m_iLowDimY, m_iLowDimX, true);
	blurFlowKernel << <m_lowGrid, m_threads2, 0, blurStream2 >> > (m_offsetArray21.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU, kernelSize, 1, m_iLowDimY, m_iLowDimX, false);

	// Synchronize streams to ensure completion
	cudaStreamSynchronize(blurStream1);
	cudaStreamSynchronize(blurStream2);

	// Clean up streams
	cudaStreamDestroy(blurStream1);
	cudaStreamDestroy(blurStream2);
}

/*
* Template instantiation
*/
template __global__ void blurFrameKernel<unsigned char>(const unsigned char* frameArray, unsigned char* blurredFrameArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char lumStart,
								const unsigned char lumEnd, const unsigned short lumPixelCount, const char chromStart, 
								const unsigned char chromEnd, const unsigned short chromPixelCount, const unsigned short dimY, const unsigned short dimX);
template __global__ void blurFrameKernel<unsigned short>(const unsigned short* frameArray, unsigned short* blurredFrameArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char lumStart,
								const unsigned char lumEnd, const unsigned short lumPixelCount, const char chromStart, 
								const unsigned char chromEnd, const unsigned short chromPixelCount, const unsigned short dimY, const unsigned short dimX);

template __global__ void calcImageDelta<unsigned char>(const unsigned char* frame1, const unsigned char* frame2, unsigned char* imageDeltaArray,
	const int* offsetArray, const unsigned int lowDimY, const unsigned int lowDimX,
	const unsigned int dimY, const unsigned int dimX, const float resolutionScalar, const unsigned int directionIdxOffset,
	const unsigned int channelIdxOffset);
template __global__ void calcImageDelta<unsigned short>(const unsigned short* frame1, const unsigned short* frame2, unsigned short* imageDeltaArray,
	const int* offsetArray, const unsigned int lowDimY, const unsigned int lowDimX,
	const unsigned int dimY, const unsigned int dimX, const float resolutionScalar, const unsigned int directionIdxOffset,
	const unsigned int channelIdxOffset);

template __global__ void calcDeltaSums8x8<unsigned char>(const unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);
template __global__ void calcDeltaSums8x8<unsigned short>(const unsigned short* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);

template __global__ void calcDeltaSums4x4<unsigned char>(const unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);
template __global__ void calcDeltaSums4x4<unsigned short>(const unsigned short* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);

template __global__ void calcDeltaSums2x2<unsigned char>(const unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);
template __global__ void calcDeltaSums2x2<unsigned short>(const unsigned short* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);

template __global__ void calcDeltaSums1x1<unsigned char>(const unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);
template __global__ void calcDeltaSums1x1<unsigned short>(const unsigned short* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);

template __global__ void artifactRemovalKernelForBlending<unsigned char>(const unsigned char* frame1, const int* hitCount, unsigned char* warpedFrame,
	const unsigned int dimY, const unsigned int dimX);
template __global__ void artifactRemovalKernelForBlending<unsigned short>(const unsigned short* frame1, const int* hitCount, unsigned short* warpedFrame,
	const unsigned int dimY, const unsigned int dimX);