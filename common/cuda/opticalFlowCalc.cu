#include <amvideo.h>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

#include <cuda_runtime_api.h>

#include "opticalFlowCalc.cuh"

// Debug message function
void CudaDebugMessage(const std::string& message) {
	const std::string m_debugMessage = message + "\n";
	OutputDebugStringA(m_debugMessage.c_str());
}

// Kernel that sets the initial offset array
__global__ void setInitialOffset(int* offsetArray, const unsigned int dimZ, const unsigned int dimY,
	const unsigned int dimX, bool firstTime) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	if (cz < dimZ && cy < dimY && cx < dimX) {
		// If it is the first time, set the default values
		if (firstTime) {
			// Set the Y direction to no offset
			offsetArray[dimZ * dimY * dimX + cz * dimY * dimX + cy * dimX + cx] = 0;

			// Set the X direction layer 0 to a -2 offset
			if (cz == 0) {
				offsetArray[cy * dimX + cx] = -2;
			// Set the X direction layer 1 to a -1 offset
			} else if (cz == 1) {
				offsetArray[dimY * dimX + cy * dimX + cx] = -1;
			// Set the X direction layer 2 to no offset
			} else if (cz == 2) {
				offsetArray[2 * dimY * dimX + cy * dimX + cx] = 0;
			// Set the X direction layer 3 to a +1 offset
			} else if (cz == 3) {
				offsetArray[3 * dimY * dimX + cy * dimX + cx] = 1;
			// Set the X direction layer 4 to a +2 offset
			} else if (cz == 4) {
				offsetArray[4 * dimY * dimX + cy * dimX + cx] = 2;
			}

		// If it is not the first time, shift the values relatively
		} else {
			const int currX = offsetArray[cy * dimX + cx];
			const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];

			// Set all Y direction layers to the previous Y direction
			offsetArray[dimZ * dimY * dimX + cz * dimY * dimX + cy * dimX + cx] = currY;

			// Shift the X direction layer 0 by -2
			if (cz == 0) {
				offsetArray[cy * dimX + cx] = currX - 2;
			// Shift the X direction layer 1 by -1
			} else if (cz == 1) {
				offsetArray[dimY * dimX + cy * dimX + cx] = currX - 1;
			// Set the X direction layer 2 to the previous X direction
			} else if (cz == 2) {
				offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX;
			// Shift the X direction layer 3 by +1
			} else if (cz == 3) {
				offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX + 1;
			// Shift the X direction layer 4 by +2
			} else if (cz == 4) {
				offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX + 2;
			}
		}
	}
}

// Kernel that calculates the absolute difference between two frames using the offset array
__global__ void calcImageDelta(const unsigned char* frame1, const unsigned char* frame2, unsigned char* imageDeltaArray,
                               const int* offsetArray, const int dimZ, const int dimY, const int dimX, const double resolutionScalar) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	if (cz < dimZ && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = -static_cast<int>(static_cast<double>(offsetArray[cz * dimY * dimX + cy * dimX + cx]));
		const int offsetY = -static_cast<int>(static_cast<double>(offsetArray[dimZ * dimY * dimX + cz * dimY * dimX + cy * dimX + cx]));

		// Current pixel is outside of frame
		if ((cy + offsetY < 0) || (cx + offsetX < 0) || (cy + offsetY > dimY) || (cx + offsetX > dimX)) {
			imageDeltaArray[cz * dimY * dimX + cy * dimX + cx] = 0;
		// Current pixel is inside of frame
		} else {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			imageDeltaArray[cz * dimY * dimX + cy * dimX + cx] = fabsf(frame1[static_cast<int>(newCy * resolutionScalar * dimX * resolutionScalar + newCx * resolutionScalar)] - frame2[static_cast<int>(cy * resolutionScalar * dimX * resolutionScalar + cx * resolutionScalar)]);
		}
	}
}

// Kernel that sums up all the pixel deltas of each window
__global__ void calcDeltaSums(unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray,
                              const unsigned int windowDimY, const unsigned int windowDimX, const unsigned int dimZ,
                              const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int windowIndexX = cx / windowDimX;
	const unsigned int windowIndexY = cy / windowDimY;

	// Check if the thread is inside the frame
	if (cz < dimZ && cy < dimY && cx < dimX) {
		atomicAdd(&summedUpDeltaArray[cz * dimY * dimX + (windowIndexY * windowDimY) * dimX + (windowIndexX * windowDimX)],
			imageDeltaArray[cz * dimY * dimX + cy * dimX + cx]);
	}
}

// Kernel that normalizes all the pixel deltas of each window
__global__ void normalizeDeltaSums(const unsigned int* summedUpDeltaArray, unsigned char* globalLowestLayerArray,
                                   const int* offsetArray, const unsigned int windowDimY, const unsigned int windowDimX,
								   const int dimZ, const int dimY, const int dimX) {
	// Allocate shared memory to share values across layers
	__shared__ float normalizedDeltaArray[5 * NUM_THREADS * NUM_THREADS];
	
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Check if the thread is a window represent
	if (cy % windowDimY == 0 && cx % windowDimX == 0) {
		// Get the current window information
		const int offsetX = offsetArray[cz * dimY * dimX + cy * dimX + cx];
		const int offsetY = offsetArray[dimZ * dimY * dimX + cz * dimY * dimX + cy * dimX + cx];

		// Calculate the number of pixels in the window
		unsigned int numPixels = windowDimY * windowDimX;

		// Calculate the not overlapping pixels
		int overlapX;
		int overlapY;

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

		const int numNotOverlappingPixels = overlapY * overlapX;
		numPixels -= numNotOverlappingPixels;

		// Normalize the summed up delta
		normalizedDeltaArray[cz * NUM_THREADS * NUM_THREADS + threadIdx.y * NUM_THREADS + threadIdx.x] = static_cast<float>(summedUpDeltaArray[cz * dimY * dimX + cy * dimX + cx]) / static_cast<
			float>(numPixels);
	}

	// Wait for all threads to finish filling the values
	__syncthreads();

	// Find the layer with the lowest value
	if (cz == 0 && cy % windowDimY == 0 && cx % windowDimX == 0) {
		unsigned char lowestLayer = 0;

		for (unsigned char z = 1; z < dimZ; ++z) {
			if (normalizedDeltaArray[z * NUM_THREADS * NUM_THREADS + threadIdx.y * NUM_THREADS + threadIdx.x] < normalizedDeltaArray[lowestLayer * NUM_THREADS * NUM_THREADS + threadIdx.y * NUM_THREADS + threadIdx.x]) {
				lowestLayer = z;
			}
		}

		globalLowestLayerArray[cy * dimX + cx] = lowestLayer;
	}
}

// Kernel that adjusts the offset array based on the comparison results
__global__ void adjustOffsetArray(int* offsetArray, const unsigned char* globalLowestLayerArray, unsigned char* statusArray,
		const unsigned int windowDimY,
		const unsigned int windowDimX, const unsigned int dimZ, const unsigned int dimY,
		const unsigned int dimX) {

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
		const unsigned int trwx = (((cx / blockDim.x) * blockDim.x) / windowDimX) * windowDimX;
		const unsigned int trwy = (((cy / blockDim.y) * blockDim.y) / windowDimY) * windowDimY;
		const unsigned int wx = (cx / windowDimX) * windowDimX;
		const unsigned int wy = (cy / windowDimY) * windowDimY;
		unsigned char lowestLayer;

		// We are the block representant
		if (threadIdx.y == 0 && threadIdx.x == 0) {
			lowestLayer = globalLowestLayerArray[wy * dimX + wx];
			lowestLayerArray[0] = lowestLayer;
		}
		
		__syncthreads();

		// We can reuse the block representants value
		if (wy == trwy && wx == trwx) {
			lowestLayer = lowestLayerArray[0];
		// The value relevant to us is different from the cached one
		} else {
			lowestLayer = globalLowestLayerArray[wy * dimX + wx];
		}

		const unsigned char currentStatus = statusArray[cy * dimX + cx];

		switch (currentStatus) {
			/*
			* X - DIRECTION
			*/
			// Find the initial x direction
			case 0:
				// If the lowest layer is 2, no x direction is needed -> continue to y direction
				if (lowestLayer == 2) {
					statusArray[cy * dimX + cx] = 3;
					const int currX = offsetArray[2 * dimY * dimX + cy * dimX + cx];
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					for (int z = 0; z < dimZ; z++) {
						offsetArray[z * dimY * dimX + cy * dimX + cx] = currX;
					}
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 1, ideal x direction found -> continue to y direction
				} else if (lowestLayer == 1) {
					statusArray[cy * dimX + cx] = 3;
					const int currX = offsetArray[dimY * dimX + cy * dimX + cx];
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[cy * dimX + cx] = currX;
					offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 3, ideal x direction found -> continue to y direction
				} else if (lowestLayer == 3) {
					statusArray[cy * dimX + cx] = 3;
					const int currX = offsetArray[3 * dimY * dimX + cy * dimX + cx];
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[cy * dimX + cx] = currX;
					offsetArray[dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[4 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 4 -> continue moving in the positive x direction
				} else if (lowestLayer == 4) {
					statusArray[cy * dimX + cx] = 1;
					const int currX = offsetArray[4 * dimY * dimX + cy * dimX + cx];
					offsetArray[0 * dimY * dimX + cy * dimX + cx] = currX + 4;
					offsetArray[1 * dimY * dimX + cy * dimX + cx] = currX + 3;
					offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX + 2;
					offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX + 1;

				// If the lowest layer is 0 -> continue moving in the negative x direction
				} else if (lowestLayer == 0) {
					statusArray[cy * dimX + cx] = 2;
					const int currX = offsetArray[cy * dimX + cx];
					offsetArray[1 * dimY * dimX + cy * dimX + cx] = currX - 1;
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
					int idealX = offsetArray[lowestLayer * dimY * dimX + cy * dimX + cx];
					for (int z = 0; z < dimZ; z++) {
						offsetArray[z * dimY * dimX + cy * dimX + cx] = idealX;
					}
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 0 -> continue moving in x direction
				} else {
					const int currX = offsetArray[cy * dimX + cx];
					offsetArray[0 * dimY * dimX + cy * dimX + cx] = currX + 4;
					offsetArray[1 * dimY * dimX + cy * dimX + cx] = currX + 3;
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
					int idealX = offsetArray[lowestLayer * dimY * dimX + cy * dimX + cx];
					for (int z = 0; z < dimZ; z++) {
						offsetArray[z * dimY * dimX + cy * dimX + cx] = idealX;
					}
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 4 -> continue moving in x direction
				} else {
					int currX = offsetArray[4 * dimY * dimX + cy * dimX + cx];
					offsetArray[0 * dimY * dimX + cy * dimX + cx] = currX;
					offsetArray[1 * dimY * dimX + cy * dimX + cx] = currX - 1;
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
				// If the lowest layer is 1,2, or 3, no y direction is needed -> we are done
				if (lowestLayer == 1 || lowestLayer == 2 || lowestLayer == 3) {
					statusArray[cy * dimX + cx] = 6;
					const int currY = offsetArray[dimZ * dimY * dimX + lowestLayer * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY;

				// If the lowest layer is 4 -> continue moving in the positive y direction
				} else if (lowestLayer == 4) {
					statusArray[cy * dimX + cx] = 4;
					const int currY = offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY + 4;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY + 3;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY + 2;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;

				// If the lowest layer is 0 -> continue moving in the negative y direction
				} else if (lowestLayer == 0) {
					statusArray[cy * dimX + cx] = 5;
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
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
					int idealY = offsetArray[dimZ * dimY * dimX + lowestLayer * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = idealY;

				// If the lowest layer is 0 -> continue moving in y direction
				} else {
					int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
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
					int idealY = offsetArray[dimZ * dimY * dimX + lowestLayer * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = idealY;

				// If the lowest layer is 4 -> continue moving in y direction
				} else {
					int currY = offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx];
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

// Kernel that warps frame1 according to the offset array
__global__ void warpFrameKernel(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones,
                                unsigned char* warpedFrame, const double frameScalar, const int dimY, const int dimX, const double resolutionScalar, const double resolutionDivider) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = static_cast<int>(static_cast<double>(offsetArray[static_cast<int>(cy * resolutionDivider * dimX * resolutionDivider + cx * resolutionDivider)]) * frameScalar) * resolutionScalar;
		const int offsetY = static_cast<int>(static_cast<double>(offsetArray[static_cast<int>(dimY * resolutionDivider * dimX * resolutionDivider + cy * resolutionDivider * dimX * resolutionDivider + cx * resolutionDivider)]) * frameScalar) * resolutionScalar;

		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			warpedFrame[newCy * dimX + newCx] = frame1[cy * dimX + cx];
			atomicAdd(&hitCount[newCy * dimX + newCx], ones[cy * dimX + cx]);
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<double>(offsetArray[static_cast<int>(2 * cy * resolutionDivider * dimX * resolutionDivider + ((cx * resolutionDivider) / 2) * 2)]) * frameScalar) * resolutionScalar;
		const int offsetY = static_cast<int>((static_cast<double>(offsetArray[static_cast<int>(dimY * resolutionDivider * dimX * resolutionDivider + 2 * cy * dimX * resolutionDivider + ((cx * resolutionDivider) / 2) * 2)]) * frameScalar) / 2.0) * resolutionScalar;

		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY / 2) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), (dimY / 2) - 1);

			// U Channel
			if (cx % 2 == 0) {
				warpedFrame[dimY * dimX + newCy * dimX + (newCx / 2) * 2] = frame1[dimY * dimX + cy * dimX + (cx / 2) * 2];

			// V Channel
			} else {
				warpedFrame[dimY * dimX + newCy * dimX + (newCx / 2) * 2 + 1] = frame1[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1];
			}
		}
	}
}

// Kernel that removes artifacts from the warped frame
__global__ void artifactRemovalKernel(const unsigned char* frame1, const int* hitCount, unsigned char* warpedFrame,
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
			// U Channel
			if (cx % 2 == 0) {
				warpedFrame[dimY * dimX + cy * dimX + (cx / 2) * 2] = frame1[dimY * dimX + cy * dimX + (cx / 2) * 2];

			// V Channel
			} else {
				warpedFrame[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1] = frame1[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1];
			}
		}
	}
}

// Kernel that blends frame1 to frame2
__global__ void blendFrameKernel(const unsigned char* frame1, const unsigned char* frame2, unsigned char* blendedFrame,
                                 const double frame1Scalar, const double frame2Scalar, const unsigned int dimY,
                                 const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Check if result is within matrix boundaries
	if ((cz == 0 && cy < dimY && cx < dimX) || (cz == 1 && cy < (dimY / 2) && cx < dimX)) {
		blendedFrame[cz * dimY * dimX + cy * dimX + cx] = static_cast<unsigned char>(static_cast<double>(frame1[cz * dimY * dimX + cy * dimX + cx]) * 
			frame1Scalar + static_cast<double>(frame2[cz * dimY * dimX + cy * dimX + cx]) * frame2Scalar);
	}
}

// Kernel that creates an HSV flow image from the offset array
__global__ void convertFlowToHSVKernel(const int* flowArray, unsigned char* nv12Array,
                                       const unsigned int dimY, const unsigned int dimX, const double saturation,
                                       const double value) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Get the current flow values
	double x = 0.0;
	double y = 0.0;
	if (cz == 0) {
		x = flowArray[cy * dimX + cx];
		y = flowArray[dimY * dimX + cy * dimX + cx];
	} else {
		x = flowArray[cy * 2 * dimX + (cx / 2) * 2];
		y = flowArray[dimY * dimX + cy * 2 * dimX + (cx / 2) * 2];
	}

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
	case 0: rgb = {static_cast<int>(value * 255), static_cast<int>(t * 255), static_cast<int>(p * 255)};
		break;
	case 1: rgb = {static_cast<int>(q * 255), static_cast<int>(value * 255), static_cast<int>(p * 255)};
		break;
	case 2: rgb = {static_cast<int>(p * 255), static_cast<int>(value * 255), static_cast<int>(t * 255)};
		break;
	case 3: rgb = {static_cast<int>(p * 255), static_cast<int>(q * 255), static_cast<int>(value * 255)};
		break;
	case 4: rgb = {static_cast<int>(t * 255), static_cast<int>(p * 255), static_cast<int>(value * 255)};
		break;
	case 5: rgb = {static_cast<int>(value * 255), static_cast<int>(p * 255), static_cast<int>(q * 255)};
		break;
	default: rgb = {0, 0, 0};
		break;
	}

	// Write the RGB values to the array
	if (cz < 2 && cy < dimY && cx < dimX) {
		// Y Channel
		if (cz == 0) {
			nv12Array[cy * dimX + cx] = static_cast<unsigned char>(fmaxf(fminf(0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b, 255), 0));
		// U Channel
		} else if (cz == 1 && cx % 2 == 0 && cy < dimY / 2) {
			nv12Array[dimY * dimX + cy * dimX + (cx / 2) * 2] = static_cast<unsigned char>(fmaxf(fminf(-0.147 * rgb.r - 0.289 * rgb.g + 0.436 * rgb.b, 255), 0));
		// V Channel
		} else if (cy < dimY / 2) {
			nv12Array[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1] = static_cast<unsigned char>(fmaxf(fminf(0.615 * rgb.r - 0.515 * rgb.g - 0.100 * rgb.b, 255), 0));
		}
	}
}

// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__global__ void flipFlowKernel(const int* flowArray12, int* flowArray21, const unsigned int dimZ,
                               const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const int cx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	const int cy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const int cz = static_cast<int>(threadIdx.z);

	// Get the current flow values
	const int x = flowArray12[cy * dimX + cx];
	const int y = flowArray12[dimZ * dimY * dimX + cy * dimX + cx];

	// Project the flow values onto the flow array from frame 2 to frame 1
	if (cy < static_cast<int>(dimY) && cx < static_cast<int>(dimX)) {
		if (cz == 0 && (cy + y) < static_cast<int>(dimY) && (cy + y) >= 0 && (cx + x) < static_cast<int>(dimX) && (cx +
			x) >= 0) {
			flowArray21[(cy + y) * dimX + cx + x] = -x;
		} else if (cz == 1 && (cy + y) < static_cast<int>(dimY) && (cy + y) >= 0 && (cx + x) < static_cast<int>(dimX) &&
			(cx + x) >= 0) {
			flowArray21[dimY * dimX + (cy + y) * dimX + cx + x] = -y;
		}
	}
}

// Kernel that blurs a flow array
__global__ void blurKernel(const int* flowArray, int* blurredFlowArray, const int kernelSize, const int dimZ, const int dimY, const int dimX) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	if (kernelSize > 1) {
		// Calculate the x and y boundaries of the kernel
		const int start = -(kernelSize / 2);
		const int end = (kernelSize / 2);

		// Collect the sum of the surrounding pixels
		if (cz < 2 && cy < dimY && cx < dimX) {
			for (int y = start; y < end; y++) {
				for (int x = start; x < end; x++) {
					if ((cy + y) < dimY && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
						atomicAdd(&blurredFlowArray[cz * dimY * dimX + cy * dimX + cx], flowArray[cz * dimZ * dimY * dimX + (cy + y) * dimX + cx + x]);
					}
				}
			}
			blurredFlowArray[cz * dimY * dimX + cy * dimX + cx] /= (end - start) * (end - start);
		}
	} else {
		if (cz < 2 && cy < dimY && cx < dimX) {
			blurredFlowArray[cz * dimY * dimX + cy * dimX + cx] = flowArray[cz * dimZ * dimY * dimX + cy * dimX + cx];
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
* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
* @param resolutionDivider: The scalar to divide the resolution with
*/
void OpticalFlowCalc::init(const unsigned int dimY, const unsigned int dimX, const double dDimScalar, const double resolutionDivider) {
	const unsigned int lowDimY = dimY * resolutionDivider;
	const unsigned int lowDimX = dimX * resolutionDivider;

	grid.x = fmaxf(ceilf(lowDimX / static_cast<float>(NUM_THREADS)), 1);
	grid.y = fmaxf(ceilf(lowDimY / static_cast<float>(NUM_THREADS)), 1);
	grid.z = 1;
	threads5.x = NUM_THREADS;
	threads5.y = NUM_THREADS;
	threads5.z = 5;
	threads2.x = NUM_THREADS;
	threads2.y = NUM_THREADS;
	threads2.z = 2;
	threads1.x = NUM_THREADS;
	threads1.y = NUM_THREADS;
	threads1.z = 1;
	highGrid.x = fmaxf(ceilf(dimX / static_cast<float>(NUM_THREADS)), 1);
	highGrid.y = fmaxf(ceilf(dimY / static_cast<float>(NUM_THREADS)), 1);
	m_frame1.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_frame2.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_imageDeltaArray.init({5, lowDimY, lowDimX});
	m_offsetArray12.init({2, 5, lowDimY, lowDimX});
	m_offsetArray21.init({2, lowDimY, lowDimX});
	m_blurredOffsetArray12.init({2, lowDimY, lowDimX});
	m_blurredOffsetArray21.init({2, lowDimY, lowDimX});
	m_statusArray.init({lowDimY, lowDimX});
	m_summedUpDeltaArray.init({5, lowDimY, lowDimX});
	m_normalizedDeltaArray.init({5, lowDimY, lowDimX});
	m_lowestLayerArray.init({lowDimY, lowDimX});
	m_warpedFrame12.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_warpedFrame21.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_blendedFrame.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_outputFrame.init({1, dimY, dimX}, 0, 3 * dimY * dimX * dDimScalar);
	m_hitCount.init({1, dimY, dimX});
	m_ones.init({1, dimY, dimX}, 1);
	m_iWindowDimX = dimX;
	m_iWindowDimY = dimY;
	m_bIsInitialized = true;
}

/*
* Returns whether the optical flow calculation is initialized
*
* @return: True if the optical flow calculation is initialized, false otherwise
*/
bool OpticalFlowCalc::isInitialized() const {
	return m_bIsInitialized;
}

/*
* Updates the frame1 array
*/
void OpticalFlowCalc::updateFrame1(const BYTE* pInBuffer) {
	m_frame1.fillData(pInBuffer);
	m_bBisNewest = false;
}

/*
* Updates the frame2 array
*/
void OpticalFlowCalc::updateFrame2(const BYTE* pInBuffer) {
	m_frame2.fillData(pInBuffer);
	m_bBisNewest = true;
}

/*
* Calculates the optical flow between frame1 and frame2
*
* @param iNumIterations: Number of iterations to calculate the optical flow
* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
* @param resolutionScalar: The scalar to scale the resolution with
*/
void OpticalFlowCalc::calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps, const double resolutionScalar) {
	// Reset variables
	m_iWindowDimX = m_imageDeltaArray.dimX;
	m_iWindowDimY = m_imageDeltaArray.dimY;
	if (iNumIterations == 0 || iNumIterations > ceil(log2f(m_imageDeltaArray.dimX))) {
		iNumIterations = ceil(log2f(m_imageDeltaArray.dimX));
	}

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		// Set the starting offset for the current window size
		setInitialOffset << <grid, threads5 >> > (m_offsetArray12.arrayPtrGPU,
												  m_imageDeltaArray.dimZ, m_imageDeltaArray.dimY,
												  m_imageDeltaArray.dimX, !iter);

		// Each step we adjust the offset array to find the ideal offset
		for (unsigned int step = 0; step < iNumSteps; step++) {
			// Reset the summed up delta array
			m_summedUpDeltaArray.fill(0);

			// 1. Calculate the image deltas with the current offset array
			if (m_bBisNewest) {
				calcImageDelta << <grid, threads5 >> >(m_frame1.arrayPtrGPU, m_frame2.arrayPtrGPU,
				                                       m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
													   m_imageDeltaArray.dimZ, m_imageDeltaArray.dimY, 
													   m_imageDeltaArray.dimX, resolutionScalar);
			} else {
				calcImageDelta << <grid, threads5 >> >(m_frame2.arrayPtrGPU, m_frame1.arrayPtrGPU,
				                                       m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
												       m_imageDeltaArray.dimZ, m_imageDeltaArray.dimY,
													   m_imageDeltaArray.dimX, resolutionScalar);
			}

			// 2. Sum up the deltas of each window
			calcDeltaSums << <grid, threads5 >> >(m_imageDeltaArray.arrayPtrGPU, m_summedUpDeltaArray.arrayPtrGPU,
			                                      m_iWindowDimY, m_iWindowDimX, m_imageDeltaArray.dimZ, 
											      m_imageDeltaArray.dimY, m_imageDeltaArray.dimX);

			// 3. Normalize the summed up delta array and find the best layer
			normalizeDeltaSums << <grid, threads5 >> >(m_summedUpDeltaArray.arrayPtrGPU, m_lowestLayerArray.arrayPtrGPU,
				                                       m_offsetArray12.arrayPtrGPU, m_iWindowDimY, m_iWindowDimX,
													   m_imageDeltaArray.dimZ, m_imageDeltaArray.dimY, 
													   m_imageDeltaArray.dimX);

			// 4. Adjust the offset array based on the comparison results
			if (true) {
				adjustOffsetArray << <grid, threads1 >> > (m_offsetArray12.arrayPtrGPU,
					m_lowestLayerArray.arrayPtrGPU, m_statusArray.arrayPtrGPU,
					m_iWindowDimY, m_iWindowDimX,
					m_imageDeltaArray.dimZ, m_imageDeltaArray.dimY,
					m_imageDeltaArray.dimX);
			}

			// Wait for all threads to finish
			//cudaDeviceSynchronize();
		}

		// 5. Adjust window size
		m_iWindowDimX = fmax(m_iWindowDimX / 2, 1);
		m_iWindowDimY = fmax(m_iWindowDimY / 2, 1);

		// Reset the status array
		m_statusArray.fill(0);
	}

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Warps frame1 according to the offset array to frame2
*
* @param offsetArray: The array containing the offsets
* @param resolutionScalar: The scalar to scale the resolution with
* @param resolutionDivider: The scalar to divide the resolution with
*/
void OpticalFlowCalc::warpFrame12(double dScalar, const double resolutionScalar, const double resolutionDivider) {
	// Calculate the blend scalar
	const double frameScalar = dScalar;

	// Reset the hit count array
	m_hitCount.fill(0);

	// Warp the frame
	if (m_bBisNewest) {
		warpFrameKernel << <highGrid, threads2 >> >(m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
		                                            m_hitCount.arrayPtrGPU, m_ones.arrayPtrGPU, m_warpedFrame12.arrayPtrGPU,
		                                            frameScalar, m_frame1.dimY, m_frame1.dimX, resolutionScalar, resolutionDivider);
		artifactRemovalKernel << <highGrid, threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount.arrayPtrGPU, m_warpedFrame12.arrayPtrGPU, m_frame1.dimY, m_frame1.dimX);
	} else {
		warpFrameKernel << <highGrid, threads2 >> >(m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
		                                            m_hitCount.arrayPtrGPU, m_ones.arrayPtrGPU, m_warpedFrame12.arrayPtrGPU,
		                                            frameScalar, m_frame1.dimY, m_frame1.dimX, resolutionScalar, resolutionDivider);
		artifactRemovalKernel << <highGrid, threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount.arrayPtrGPU, m_warpedFrame12.arrayPtrGPU, m_frame1.dimY, m_frame1.dimX);
	}

	// Wait for all threads to finish
	//cudaDeviceSynchronize();

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Warps frame2 according to the offset array to frame1
*
* @param offsetArray: The array containing the offsets
* @param resolutionScalar: The scalar to scale the resolution with
* @param resolutionDivider: The scalar to divide the resolution with
*/
void OpticalFlowCalc::warpFrame21(double dScalar, const double resolutionScalar, const double resolutionDivider) {
	// Calculate the blend scalar
	const double frameScalar = 1.0 - dScalar;

	// Reset the hit count array
	m_hitCount.fill(0);

	// Warp the frame
	if (m_bBisNewest) {
		warpFrameKernel << <highGrid, threads2 >> >(m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
		                                            m_hitCount.arrayPtrGPU, m_ones.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
		                                            frameScalar, m_frame1.dimY, m_frame1.dimX, resolutionScalar, resolutionDivider);
		artifactRemovalKernel << <highGrid, threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU, m_frame2.dimY, m_frame2.dimX);
	} else {
		warpFrameKernel << <highGrid, threads2 >> >(m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
		                                            m_hitCount.arrayPtrGPU, m_ones.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
		                                            frameScalar, m_frame1.dimY, m_frame1.dimX, resolutionScalar, resolutionDivider);
		artifactRemovalKernel << <highGrid, threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU, m_frame2.dimY, m_frame2.dimX);
	}

	// Wait for all threads to finish
	//cudaDeviceSynchronize();

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
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
	blendFrameKernel << <highGrid, threads2 >> >(m_warpedFrame12.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
	                                         m_blendedFrame.arrayPtrGPU, frame1Scalar, frame2Scalar,
	                                         m_warpedFrame12.dimY, m_warpedFrame12.dimX);

	// Wait for all threads to finish
	//cudaDeviceSynchronize();

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Draws the flow as an RGB image
*
* @param saturation: The saturation of the flow image
* @param value: The value of the flow image
*/
void OpticalFlowCalc::drawFlowAsHSV(const double saturation, const double value) const {
	// Launch kernel
	convertFlowToHSVKernel << <grid, threads2 >> >(m_blurredOffsetArray12.arrayPtrGPU, m_blendedFrame.arrayPtrGPU,
	                                               m_offsetArray12.dimY, m_offsetArray12.dimX, saturation, value);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
*/
void OpticalFlowCalc::flipFlow() {
	// Reset the offset array
	m_offsetArray21.fill(0);

	// Launch kernel
	flipFlowKernel << <grid, threads2 >> >(m_offsetArray12.arrayPtrGPU, m_offsetArray21.arrayPtrGPU,
										   m_imageDeltaArray.dimZ, m_offsetArray12.dimY, m_offsetArray12.dimX);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Blurs the offset arrays
*
* @param kernelSize: Size of the kernel to use for the blur
*/
void OpticalFlowCalc::blurFlowArrays(int kernelSize) {
	// Reset the blurred arrays
	m_blurredOffsetArray12.fill(0);
	m_blurredOffsetArray21.fill(0);

	// Launch kernels
	blurKernel << <grid, threads2 >> > (m_offsetArray12.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU, kernelSize, m_imageDeltaArray.dimZ, m_blurredOffsetArray12.dimY, m_blurredOffsetArray12.dimX);
	blurKernel << <grid, threads2 >> > (m_offsetArray21.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU, kernelSize, 1, m_blurredOffsetArray21.dimY, m_blurredOffsetArray21.dimX);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}