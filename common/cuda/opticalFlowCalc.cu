#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include "opticalFlowCalc.cuh"

// Kernel that sets the initial offset array
__global__ void setInitialOffset(int* offsetArray, const unsigned int numLayers, const unsigned int lowDimY, 
								 const unsigned int lowDimX, const unsigned int layerIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z;

	if (cy < lowDimY && cx < lowDimX) {
		switch (cz) {
			// Set the X direction layer 1 to a -2 offset
			case 0:
				offsetArray[layerIdxOffset + cy * lowDimX + cx] = -2;
				return;
			// Set the X direction layer 2 to a -1 offset
			case 1:
				offsetArray[2 * layerIdxOffset + cy * lowDimX + cx] = -1;
				return;
			// Set the X direction layer 3 to a +1 offset
			case 2:
				offsetArray[3 * layerIdxOffset + cy * lowDimX + cx] = 1;
				return;
			// Set the X direction layer 4 to a +2 offset
			case 3:
				offsetArray[4 * layerIdxOffset + cy * lowDimX + cx] = 2;
				return;
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

	const bool isYChannel = threadIdx.z == 0 && cy < lowDimY && cx < lowDimX;
	const bool isUVChannel = cy < (lowDimY >> 1) && cx < lowDimX;

	// Is the current thread supposed to perform calculations
	if (isYChannel || isUVChannel) {
		const unsigned int layerOffset = blockIdx.z * lowDimY * lowDimX; // Offset to index the layer of the current thread
		const unsigned int scaledCx = static_cast<unsigned int>(static_cast<float>(cx) * resolutionScalar); // The X-Index of the current thread in the input frames
		const unsigned int scaledCy = static_cast<unsigned int>(static_cast<float>(cy) * resolutionScalar); // The Y-Index of the current thread in the input frames

		const unsigned int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim
		const unsigned int threadIndex3D = cz * lowDimY * lowDimX + threadIndex2D; // Standard thread index

		// Y-Channel
		if (isYChannel) {
			const int offsetX = -offsetArray[layerOffset + threadIndex2D];
			const int offsetY = -offsetArray[directionIdxOffset + layerOffset + threadIndex2D];
			const int newCx = scaledCx + offsetX;
			const int newCy = scaledCy + offsetY;

			imageDeltaArray[threadIndex3D] = (newCy < 0 || newCx < 0 || newCy >= dimY || newCx >= dimX) ? 0 : 
				abs(frame1[newCy * dimX + newCx] - frame2[scaledCy * dimX + scaledCx]);


		// U/V-Channel
		} else if (isUVChannel) {
			const unsigned int evenCx = cx & ~1; // The X-Index of the current thread rounded to be even
			const int offsetX = -offsetArray[layerOffset + (cy << 1) * lowDimX + evenCx];
			const int offsetY = -offsetArray[directionIdxOffset + layerOffset + (cy << 1) * lowDimX + evenCx];
			const int newCx = scaledCx + (offsetX & ~1);
			const int newCy = scaledCy + (offsetY >> 1);

			imageDeltaArray[threadIndex3D] = (newCy < 0 || newCx < 0 || newCy >= (dimY >> 1) || newCx >= dimX) ? 0 : 
				abs(frame1[channelIdxOffset + newCy * dimX + newCx] - frame2[channelIdxOffset + scaledCy * dimX + scaledCx]) << 1;
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
	const unsigned int chromX = cx & ~1;
	const unsigned int chromY = cy >> 1;
	partial_sums[tIdx] = 0;

	if (cy < lowDimY && cx < lowDimX) {
		// Add the luminace and color channels together
		partial_sums[tIdx] = 
			imageDeltaArray[cz * layerIdxOffset + cy * lowDimX + cx] + 
			imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + chromY * lowDimX + chromX] +
			imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + chromY * lowDimX + chromX + 1];
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
	const unsigned int chromX = cx & ~1;
	const unsigned int chromY = cy >> 1;
	partial_sums[tIdx] = 0;

	if (cy < lowDimY && cx < lowDimX) {
		// Add the luminace and color channels together
		partial_sums[tIdx] = 
			imageDeltaArray[cz * layerIdxOffset + cy * lowDimX + cx] + 
			imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + chromY * lowDimX + chromX] +
			imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + chromY * lowDimX + chromX + 1];
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
	const unsigned int chromX = cx & ~1;
	const unsigned int chromY = cy >> 1;
	partial_sums[tIdx] = 0;

	if (cy < lowDimY && cx < lowDimX) {
		// Add the luminace and color channels together
		partial_sums[tIdx] = 
			imageDeltaArray[cz * layerIdxOffset + cy * lowDimX + cx] + 
			imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + chromY * lowDimX + chromX] +
			imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + chromY * lowDimX + chromX + 1];
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
	const unsigned int chromX = cx & ~1;
	const unsigned int chromY = cy >> 1;

	if (cy < lowDimY && cx < lowDimX) {
		// Add the luminace and color channels together
		summedUpDeltaArray[cz * channelIdxOffset + cy * lowDimX + cx] = 
			imageDeltaArray[cz * layerIdxOffset + cy * lowDimX + cx] + 
			imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + chromY * lowDimX + chromX] +
			imageDeltaArray[cz * layerIdxOffset + channelIdxOffset + chromY * lowDimX + chromX + 1];
	}
}

// Kernel that normalizes all the pixel deltas of each window
__global__ void normalizeDeltaSums(const unsigned int* summedUpDeltaArray, unsigned char* globalLowestLayerArray,
                                   const int* offsetArray, const unsigned int windowDim, int numPixels,
								   const unsigned int directionIdxOffset, const unsigned int layerIdxOffset, 
								   const unsigned int numLayers, const unsigned int lowDimY, const unsigned int lowDimX) {
	// Allocate shared memory to share values across layers
	__shared__ float normalizedDeltaArray[5 * 8 * 8 * 4];
	
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim
	bool isWindowRepresent = cy % windowDim == 0 && cx % windowDim == 0;

	// Check if the thread is a window represent
	if (isWindowRepresent) {
		// Get the current window information
		const int offsetX = offsetArray[cz * layerIdxOffset + threadIndex2D];
		const int offsetY = offsetArray[directionIdxOffset + cz * layerIdxOffset + threadIndex2D];

		// Calculate the not overlapping pixels
		int overlapX;
		int overlapY;

		// Calculate the number of not overlapping pixels
		if ((cx + windowDim + abs(offsetX) > lowDimX) || (cx - offsetX < 0)) {
			overlapX = abs(offsetX);
		} else {
			overlapX = 0;
		}

		if ((cy + windowDim + abs(offsetY) > lowDimY) || (cy - offsetY < 0)) {
			overlapY = abs(offsetY);
		} else {
			overlapY = 0;
		}

		const int numNotOverlappingPixels = overlapY * overlapX;
		numPixels -= numNotOverlappingPixels;
		numPixels = max(numPixels, 1);

		// Normalize the summed up delta
		normalizedDeltaArray[cz * 8 * 8 + threadIdx.y * 8 + threadIdx.x] = static_cast<float>(summedUpDeltaArray[cz * layerIdxOffset + threadIndex2D]) / static_cast<float>(numPixels);
	}

	// Wait for all threads to finish filling the values
	__syncthreads();

	// Find the layer with the lowest value
	if (cz == 0 && isWindowRepresent) {
		unsigned char lowestLayer = 0;

		for (unsigned char z = 1; z < numLayers; ++z) {
			if (normalizedDeltaArray[z * 8 * 8 + threadIdx.y * 8 + threadIdx.x] < 
				normalizedDeltaArray[lowestLayer * 8 * 8 + threadIdx.y * 8 + threadIdx.x]) {
				lowestLayer = z;
			}
		}

		globalLowestLayerArray[threadIndex2D] = lowestLayer;
	}
}

// Kernel that adjusts the offset array based on the comparison results
__global__ void adjustOffsetArray(int* offsetArray, const unsigned char* globalLowestLayerArray, unsigned char* statusArray,
								  const unsigned int windowDim, const unsigned int directionIdxOffset, const unsigned int layerIdxOffset, 
								  const unsigned int numLayers, const unsigned int lowDimY, const unsigned int lowDimX, const bool lastRun) {

	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim

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

	if (cy < lowDimY && cx < lowDimX) {
		const unsigned char currentStatus = statusArray[threadIndex2D];

		// We are done searching and we only need to do cleanup on the last run, so we exit here
		if (currentStatus == 6 && !lastRun) {
			return;
		}

		// We only need the lowestLayer if we are still searching
		unsigned char lowestLayer = 0;
		if (currentStatus != 6) {
			const unsigned int wx = (cx / windowDim) * windowDim;
			const unsigned int wy = (cy / windowDim) * windowDim;
			lowestLayer = globalLowestLayerArray[wy * lowDimX + wx];
		}

		int currX;
		int currY;

		// If this is the last run, we need to adjust the offset array for the next iteration
		if (lastRun) {
			switch (currentStatus) {
				// We are still trying to find the perfect x direction
				case 0:
				case 1:
				case 2:
					currX = offsetArray[lowestLayer * layerIdxOffset + threadIndex2D];

					// Shift the X direction layer 0 to the ideal X direction
					offsetArray[threadIndex2D] = currX;
					// Shift the X direction layer 1 by -2
					offsetArray[layerIdxOffset + threadIndex2D] = currX - 2;
					// Shift the X direction layer 2 by -1
					offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 1;
					// Shift the X direction layer 3 by +1
					offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
					// Shift the X direction layer 4 by +2
					offsetArray[4 * layerIdxOffset + threadIndex2D] = currX + 2;
					return;

				// We are still trying to find the perfect y direction
				case 3:
				case 4:
				case 5:
					currX = offsetArray[threadIndex2D];
					currY = offsetArray[directionIdxOffset + lowestLayer * layerIdxOffset + threadIndex2D];

					// Set all Y direction layers to the ideal Y direction
					for (unsigned int z = 0; z < numLayers; z++) {
						offsetArray[directionIdxOffset + z * layerIdxOffset + threadIndex2D] = currY;
					}

					// Shift the X direction layer 1 by -2
					offsetArray[layerIdxOffset + threadIndex2D] = currX - 2;
					// Shift the X direction layer 2 by -1
					offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 1;
					// Shift the X direction layer 3 by +1
					offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
					// Shift the X direction layer 4 by +2
					offsetArray[4 * layerIdxOffset + threadIndex2D] = currX + 2;
					return;

				// Search completed, so we adjust the offset array for the next run
				default:
					currX = offsetArray[threadIndex2D];
					currY = offsetArray[directionIdxOffset + threadIndex2D];

					// Set all Y direction layers to the ideal Y direction
					for (unsigned int z = 1; z < numLayers; z++) {
						offsetArray[directionIdxOffset + z * layerIdxOffset + threadIndex2D] = currY;
					}

					// Shift the X direction layer 1 by -2
					offsetArray[layerIdxOffset + threadIndex2D] = currX - 2;
					// Shift the X direction layer 2 by -1
					offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 1;
					// Shift the X direction layer 3 by +1
					offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
					// Shift the X direction layer 4 by +2
					offsetArray[4 * layerIdxOffset + threadIndex2D] = currX + 2;
					return;
			}
			return;
		}

		// If we are still calculating, adjust the offset array based on the current status and lowest layer
		int idealX;
		int idealY;
		switch (currentStatus) {
			/*
			* X - DIRECTION
			*/
			// Find the initial x direction
			case 0:
				switch (lowestLayer) {
					// If the lowest layer is 0, no x direction is needed -> continue to y direction
					case 0:
						statusArray[threadIndex2D] = 3;
						currX = offsetArray[threadIndex2D];
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						for (int z = 0; z < numLayers; z++) {
							offsetArray[z * layerIdxOffset + threadIndex2D] = currX;
						}
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;
					
					// If the lowest layer is 1 -> continue moving in the negative x direction
					case 1:
						statusArray[threadIndex2D] = 2;
						currX = offsetArray[layerIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX;
						offsetArray[layerIdxOffset + threadIndex2D] = currX - 1;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 2;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX - 3;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX - 4;
						return;

					// If the lowest layer is 2, ideal x direction found -> continue to y direction
					case 2:
						statusArray[threadIndex2D] = 3;
						currX = offsetArray[2 * layerIdxOffset + threadIndex2D];
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX;
						offsetArray[layerIdxOffset + threadIndex2D] = currX;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;

					// If the lowest layer is 3, ideal x direction found -> continue to y direction
					case 3:
						statusArray[threadIndex2D] = 3;
						currX = offsetArray[3 * layerIdxOffset + threadIndex2D];
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX;
						offsetArray[layerIdxOffset + threadIndex2D] = currX;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;

					// If the lowest layer is 4 -> continue moving in the positive x direction
					case 4:
						statusArray[threadIndex2D] = 1;
						currX = offsetArray[4 * layerIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX + 4;
						offsetArray[layerIdxOffset + threadIndex2D] = currX + 3;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX + 2;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
						return;
				}
				return;

			// Find extended positive x direction
			case 1:
				switch (lowestLayer) {
					// If the lowest layer is 0 -> continue moving in x direction
					case 0:
						currX = offsetArray[threadIndex2D];
						offsetArray[threadIndex2D] = currX + 4;
						offsetArray[layerIdxOffset + threadIndex2D] = currX + 3;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX + 2;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX;
						return;

					// If the lowest layer is not 0, no x further direction is needed -> continue to y direction
					default:
						statusArray[threadIndex2D] = 3;
						idealX = offsetArray[lowestLayer * layerIdxOffset + threadIndex2D];
						for (unsigned int z = 0; z < numLayers; z++) {
							offsetArray[z * layerIdxOffset + threadIndex2D] = idealX;
						}
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;
				}
				return;

			// Find extended negative x direction
			case 2:
				switch (lowestLayer) {
					// If the lowest layer is not 4, no x further direction is needed -> continue to y direction
					case 0:
					case 1:
					case 2:
					case 3:
						statusArray[threadIndex2D] = 3;
						idealX = offsetArray[lowestLayer * layerIdxOffset + threadIndex2D];
						for (unsigned int z = 0; z < numLayers; z++) {
							offsetArray[z * layerIdxOffset + threadIndex2D] = idealX;
						}
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;

					// If the lowest layer is 4 -> continue moving in x direction
					case 4:
						currX = offsetArray[4 * layerIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX;
						offsetArray[layerIdxOffset + threadIndex2D] = currX - 1;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 2;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX - 3;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX - 4;
						return;
				}
				return;

			/*
			* Y - DIRECTION
			*/
			// Find the initial y direction
			case 3:
				switch (lowestLayer) {
					// If the lowest layer is 0, 2, or 3, no y direction is needed -> we are done
					case 0:
					case 2:
					case 3:
						statusArray[threadIndex2D] = 6;
						if (lowestLayer != 0) {
							currY = offsetArray[directionIdxOffset + lowestLayer * layerIdxOffset + threadIndex2D];
							offsetArray[directionIdxOffset + threadIndex2D] = currY;
						}
						return;

					// If the lowest layer is 1 -> continue moving in the negative y direction
					case 1:
						statusArray[threadIndex2D] = 5;
						currY = offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = currY;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY - 3;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY - 4;
						return;

					// If the lowest layer is 4 -> continue moving in the positive y direction
					case 4:
						statusArray[threadIndex2D] = 4;
						currY = offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = currY + 4;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY + 3;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY + 2;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						return;
				}
				return;

			// Find extended positive y direction
			case 4:
				switch (lowestLayer) {
					// If the lowest layer is 0 -> continue moving in y direction
					case 0:
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = currY + 4;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY + 3;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY + 2;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY;
						return;

					// If the lowest layer is not 0, no y further direction is needed -> we are done
					default:
						statusArray[threadIndex2D] = 6;
						idealY = offsetArray[directionIdxOffset + lowestLayer * layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = idealY;
						return;
				}
				return;

			// Find extended negative y direction
			case 5:
				switch (lowestLayer) {
					// If the lowest layer is not 4, no y further direction is needed -> we are done
					case 0:
					case 1:
					case 2:
					case 3:
						statusArray[threadIndex2D] = 6;
						idealY = offsetArray[directionIdxOffset + lowestLayer * layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = idealY;
						return;

					// If the lowest layer is 4 -> continue moving in y direction
					case 4:
						currY = offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = currY;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY - 3;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY - 4;
						return;
				}
				return;

			// Search is complete
			default:
				return;
		}
	}
}

// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__global__ void flipFlowKernel(const int* flowArray12, int* flowArray21, const int lowDimY, const int lowDimX, 
							   const float resolutionDivider, const unsigned int directionIdxOffset,
							   const unsigned int layerIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Check if we are inside the flow array
	if (cy < lowDimY && cx < lowDimX) {
		// Get the current flow values
		const int x = flowArray12[cy * lowDimX + cx];
		const int y = flowArray12[directionIdxOffset + cy * lowDimX + cx];
		const int scaledX = static_cast<int>(static_cast<float>(x) * resolutionDivider);
		const int scaledY = static_cast<int>(static_cast<float>(y) * resolutionDivider);

		// Project the flow values onto the flow array from frame 2 to frame 1
		// X-Layer
		if (cz == 0 && (cy + scaledY) < lowDimY && (cy + scaledY) >= 0 && (cx + scaledX) < lowDimX && (cx + scaledX) >= 0) {
			flowArray21[(cy + scaledY) * lowDimX + cx + scaledX] = -x;
		// Y-Layer
		} else if (cz == 1 && (cy + scaledY) < lowDimY && (cy + scaledY) >= 0 && (cx + scaledX) < lowDimX && (cx + scaledX) >= 0) {
			flowArray21[layerIdxOffset + (cy + scaledY) * lowDimX + cx + scaledX] = -y;
		}
	}
}

// Kernel that blurs a flow array
__global__ void blurFlowKernel(const int* flowArray, int* blurredFlowArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char start,
								const unsigned char end, const unsigned short pixelCount, const unsigned short numLayers,
								const unsigned short lowDimY, const unsigned short lowDimX) {
	// Shared memory for the flow to prevent multiple global memory accesses
	extern __shared__ int sharedFlowArray[];

	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = blockIdx.z;

	// Current threadblock index
	const int trX = blockIdx.x * blockDim.x;
	const int trY = blockIdx.y * blockDim.y;
	unsigned char offsetX;
	unsigned char offsetY;

    // Calculate the number of entries to fill for this thread
    const unsigned short threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned char entriesToFill = avgEntriesPerThread + (threadIndex < remainder ? 1 : 0);

    // Calculate the starting index for this thread
    unsigned short startIndex = 0;
    for (unsigned short i = 0; i < threadIndex; ++i) {
        startIndex += avgEntriesPerThread + (i < remainder ? 1 : 0);
    }

    // Fill the shared memory for this thread
    for (unsigned short i = 0; i < entriesToFill; ++i) {
		offsetX = (startIndex + i) % chacheSize;
		offsetY = (startIndex + i) / chacheSize;
		if ((trY - boundsOffset + offsetY) < lowDimY && (trX - boundsOffset + offsetX) < lowDimX) {
			sharedFlowArray[startIndex + i] = flowArray[cz * numLayers * lowDimY * lowDimX + (trY - boundsOffset + offsetY) * lowDimX + (trX - boundsOffset + offsetX)];
		} else {
			sharedFlowArray[startIndex + i] = 0;
		}
	}

    // Ensure all threads have finished loading before continuing
    __syncthreads();

	// Check if we are inside the flow array
	if (cy < lowDimY && cy >= 0 && cx < lowDimX && cx >= 0) {
		// Calculate the x and y boundaries of the kernel
		int blurredOffset = 0;

		// Collect the sum of the surrounding values
		for (char y = start; y < end; y++) {
			for (char x = start; x < end; x++) {
				if ((cy + y) < lowDimY && (cy + y) >= 0 && (cx + x) < lowDimX && (cx + x) >= 0) {
					blurredOffset += sharedFlowArray[(threadIdx.y + boundsOffset + y) * chacheSize + threadIdx.x + boundsOffset + x];
				} else {
					blurredOffset += sharedFlowArray[(threadIdx.y + boundsOffset) * chacheSize + threadIdx.x + boundsOffset];
				}
			}
		}
		blurredOffset /= pixelCount;
		blurredFlowArray[cz * lowDimY * lowDimX + cy * lowDimX + cx] = blurredOffset;
	}
	
}

// Kernel that removes artifacts from the warped frame
template <typename T>
__global__ void artifactRemovalKernelForBlending(const T* frame1, const int* hitCount, T* warpedFrame,
												 const unsigned int dimY, const unsigned int dimX, const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int threadIndex2D = cy * dimX + cx; // Standard thread index without Z-Dim

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[threadIndex2D] = frame1[threadIndex2D];
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[channelIdxOffset + threadIndex2D] = frame1[channelIdxOffset + threadIndex2D];
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
	flipFlowKernel << <m_lowGrid16x16x1, m_threads16x16x2 >> > (m_offsetArray12.arrayPtrGPU, m_offsetArray21.arrayPtrGPU,
												            m_iLowDimY, m_iLowDimX, m_fResolutionDivider, m_iDirectionIdxOffset, m_iLayerIdxOffset);
}

/*
* Blurs the offset arrays
*
* @param kernelSize: Size of the kernel to use for the blur
*/
void OpticalFlowCalc::blurFlowArrays(const int kernelSize) const {
	const unsigned char boundsOffset = kernelSize >> 1;
	const unsigned char chacheSize = kernelSize + (boundsOffset << 1);
	const size_t sharedMemSize = chacheSize * chacheSize * sizeof(int);
	const unsigned short totalThreads = max(kernelSize * kernelSize, 1);
    const unsigned short totalEntries = chacheSize * chacheSize;
    const unsigned char avgEntriesPerThread = totalEntries / totalThreads;
	const unsigned short remainder = totalEntries % totalThreads;
	const char start = -(kernelSize >> 1);
	const unsigned char end = (kernelSize >> 1);
	const unsigned short pixelCount = (end - start) * (end - start);

	// Calculate the number of blocks needed
	const unsigned int NUM_BLOCKS_X = max(static_cast<int>(ceil(static_cast<double>(m_iLowDimX) / kernelSize)), 1);
	const unsigned int NUM_BLOCKS_Y = max(static_cast<int>(ceil(static_cast<double>(m_iLowDimY) / kernelSize)), 1);

	// Use dim3 structs for block and grid size
	dim3 gridBF(NUM_BLOCKS_X, NUM_BLOCKS_Y, 2);
	dim3 threadsBF(kernelSize, kernelSize, 1);

	// No need to blur the flow if the kernel size is less than 4
	if (kernelSize < 4) {
		// Offset12 X-Dir
		cudaMemcpy(m_blurredOffsetArray12.arrayPtrGPU, m_offsetArray12.arrayPtrGPU, m_iLayerIdxOffset * sizeof(int), cudaMemcpyDeviceToDevice);
		// Offset12 Y-Dir
		cudaMemcpy(m_blurredOffsetArray12.arrayPtrGPU + m_iLayerIdxOffset, m_offsetArray12.arrayPtrGPU + m_iDirectionIdxOffset, m_iLayerIdxOffset * sizeof(int), cudaMemcpyDeviceToDevice);
		// Offset21 X&Y-Dir
		cudaMemcpy(m_blurredOffsetArray21.arrayPtrGPU, m_offsetArray21.arrayPtrGPU, m_offsetArray21.bytes, cudaMemcpyDeviceToDevice);
	} else {
		// Create CUDA streams
		cudaStream_t blurStream1, blurStream2;
		cudaStreamCreate(&blurStream1);
		cudaStreamCreate(&blurStream2);

		// Launch kernels
		blurFlowKernel << <gridBF, threadsBF, sharedMemSize, blurStream1 >> > (m_offsetArray12.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU, kernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, start, end, pixelCount, m_iNumLayers, m_iLowDimY, m_iLowDimX);
		blurFlowKernel << <gridBF, threadsBF, sharedMemSize, blurStream2 >> > (m_offsetArray21.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU, kernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, start, end, pixelCount, 1, m_iLowDimY, m_iLowDimX);

		// Synchronize streams to ensure completion
		cudaStreamSynchronize(blurStream1);
		cudaStreamSynchronize(blurStream2);

		// Clean up streams
		cudaStreamDestroy(blurStream1);
		cudaStreamDestroy(blurStream2);
	}
}

/*
* Template instantiation
*/
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
	const unsigned int dimY, const unsigned int dimX, const unsigned int channelIdxOffset);
template __global__ void artifactRemovalKernelForBlending<unsigned short>(const unsigned short* frame1, const int* hitCount, unsigned short* warpedFrame,
	const unsigned int dimY, const unsigned int dimX, const unsigned int channelIdxOffset);