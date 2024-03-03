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

// Kernel that converts an NV12 array to a P010 array
__global__ void convertNV12toP010Kernel(const unsigned char* nv12Array, unsigned short* p010Array, const unsigned int dimY, 
									    const unsigned int dimX, const double dDimScalar) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	if (cz < 2 && cy < static_cast<unsigned int>(dimY * dDimScalar) && cx < static_cast<unsigned int>(dimX * dDimScalar)) {
		if ((cz == 0 && cy < static_cast<unsigned int>(dimY * dDimScalar) && cx < static_cast<unsigned int>(dimX * dDimScalar)) || 
			(cz == 1 && cy < ((static_cast<unsigned int>(dimY * dDimScalar)) / 2) && cx < static_cast<unsigned int>(dimX * dDimScalar))) {
			p010Array[static_cast<unsigned int>(cz * dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + cx] = static_cast<unsigned short>(nv12Array[cz * dimY * dimX + cy * dimX + cx]) << 8;
		}
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
__global__ void calcImageDelta(const unsigned char* frame1, const unsigned char* frame2, unsigned char* imageDeltaArray,
							   const int* offsetArray, const int dimZ, const int dimY, const int dimX, const float fResolutionScalar) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	if (cz < dimZ && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = -offsetArray[cz * dimY * dimX + cy * dimX + cx];
		const int offsetY = -offsetArray[dimZ * dimY * dimX + cz * dimY * dimX + cy * dimX + cx];

		// Current pixel is outside of frame
		if ((cy * fResolutionScalar + offsetY < 0) || (cx * fResolutionScalar + offsetX < 0) || 
			(cy * fResolutionScalar + offsetY >= dimY * fResolutionScalar) || 
			(cx * fResolutionScalar + offsetX >= dimX * fResolutionScalar)) {
			imageDeltaArray[cz * dimY * dimX + cy * dimX + cx] = 0;
		// Current pixel is inside of frame
		} else {
			const int newCx = fminf(fmaxf(cx * fResolutionScalar + offsetX, 0), dimX * fResolutionScalar - 1);
			const int newCy = fminf(fmaxf(cy * fResolutionScalar + offsetY, 0), dimY * fResolutionScalar - 1);
			imageDeltaArray[cz * dimY * dimX + cy * dimX + cx] = abs(frame1[newCy * static_cast<unsigned int>(dimX * fResolutionScalar) + newCx] - 
				frame2[static_cast<unsigned int>(cy * fResolutionScalar) * static_cast<unsigned int>(dimX * fResolutionScalar) + static_cast<unsigned int>(cx * fResolutionScalar)]);
		}
	}
}

// Kernel that sums up all the pixel deltas of each window
__global__ void calcDeltaSums(unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int windowDimY, 
							  const unsigned int windowDimX, const unsigned int dimZ, const unsigned int dimY, const unsigned int dimX) {
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
	__shared__ double normalizedDeltaArray[5 * NUM_THREADS * NUM_THREADS];
	
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
		normalizedDeltaArray[cz * NUM_THREADS * NUM_THREADS + threadIdx.y * NUM_THREADS + threadIdx.x] = static_cast<double>(summedUpDeltaArray[cz * dimY * dimX + cy * dimX + cx]) / static_cast<double>(numPixels);
	}

	// Wait for all threads to finish filling the values
	__syncthreads();

	// Find the layer with the lowest value
	if (cz == 0 && cy % windowDimY == 0 && cx % windowDimX == 0) {
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
								  const unsigned int windowDimY, const unsigned int windowDimX, const unsigned int dimZ, 
								  const unsigned int dimY, const unsigned int dimX) {

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

		switch (currentStatus) {
			/*
			* X - DIRECTION
			*/
			// Find the initial x direction
			case 0:
				// If the lowest layer is 0, no x direction is needed -> continue to y direction
				if (lowestLayer == 0) {
					statusArray[cy * dimX + cx] = 3;
					const int currX = offsetArray[cy * dimX + cx];
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
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
					const int currX = offsetArray[2 * dimY * dimX + cy * dimX + cx];
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
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
					const int currX = offsetArray[3 * dimY * dimX + cy * dimX + cx];
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
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
					const int currX = offsetArray[4 * dimY * dimX + cy * dimX + cx];
					offsetArray[cy * dimX + cx] = currX + 4;
					offsetArray[dimY * dimX + cy * dimX + cx] = currX + 3;
					offsetArray[2 * dimY * dimX + cy * dimX + cx] = currX + 2;
					offsetArray[3 * dimY * dimX + cy * dimX + cx] = currX + 1;

				// If the lowest layer is 1 -> continue moving in the negative x direction
				} else if (lowestLayer == 1) {
					statusArray[cy * dimX + cx] = 2;
					const int currX = offsetArray[dimY * dimX + cy * dimX + cx];
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
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 0 -> continue moving in x direction
				} else {
					const int currX = offsetArray[cy * dimX + cx];
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
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY + 2;

				// If the lowest layer is 4 -> continue moving in x direction
				} else {
					const int currX = offsetArray[4 * dimY * dimX + cy * dimX + cx];
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
						const int currY = offsetArray[dimZ * dimY * dimX + lowestLayer * dimY * dimX + cy * dimX + cx];
						offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY;
					}

				// If the lowest layer is 4 -> continue moving in the positive y direction
				} else if (lowestLayer == 4) {
					statusArray[cy * dimX + cx] = 4;
					const int currY = offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY + 4;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY + 3;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY + 2;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY + 1;

				// If the lowest layer is 1 -> continue moving in the negative y direction
				} else if (lowestLayer == 1) {
					statusArray[cy * dimX + cx] = 5;
					const int currY = offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx];
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
					const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];
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
					const int currY = offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx];
					offsetArray[dimZ * dimY * dimX + cy * dimX + cx] = currY;
					offsetArray[dimZ * dimY * dimX + dimY * dimX + cy * dimX + cx] = currY - 1;
					offsetArray[dimZ * dimY * dimX + 2 * dimY * dimX + cy * dimX + cx] = currY - 2;
					offsetArray[dimZ * dimY * dimX + 3 * dimY * dimX + cy * dimX + cx] = currY - 3;
					offsetArray[dimZ * dimY * dimX + 4 * dimY * dimX + cy * dimX + cx] = currY - 4;
				}
				break;
				
			// Adjust offset for next iteration
			case 6:
				statusArray[cy * dimX + cx] = 7;
				const int currX = offsetArray[cy * dimX + cx];
				const int currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];

				// Set all Y direction layers to the previous Y direction
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

			// Search is complete
			default:
				break;
		}
	}
}

// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__global__ void flipFlowKernel(const int* flowArray12, int* flowArray21, const unsigned int dimZ,
							   const int dimY, const int dimX, const float fResolutionDivider) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Check if we are inside the flow array
	if (cz < 2 && cy < dimY && cx < dimX) {
		// Get the current flow values
		const int x = flowArray12[cy * dimX + cx];
		const int y = flowArray12[dimZ * dimY * dimX + cy * dimX + cx];
		const int scaledX = static_cast<int>(x * fResolutionDivider);
		const int scaledY = static_cast<int>(y * fResolutionDivider);

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
__global__ void blurKernel(const int* flowArray, int* blurredFlowArray, const int kernelSize, const int dimZ, const int dimY, 
						   const int dimX, const bool offset12) {
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
			if (offset12) {
				blurredFlowArray[cz * dimY * dimX + cy * dimX + cx] = flowArray[cz * dimZ * dimY * dimX + cy * dimX + cx];
			} else {
				blurredFlowArray[cz * dimY * dimX + cy * dimX + cx] = flowArray[cz * dimY * dimX + cy * dimX + cx];
			}
		}
	}
}

// Kernel that warps a frame according to the offset array
__global__ void warpFrameKernelForOutput(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones,
									     unsigned short* warpedFrame, const float frameScalar, const int dimY, const int dimX, 
									     const float fResolutionDivider, const double dDimScalar) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>(cx * fResolutionDivider)]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(dimY * fResolutionDivider * dimX * fResolutionDivider) + static_cast<unsigned int>(cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>(cx * fResolutionDivider)]) * frameScalar);

		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			warpedFrame[newCy * static_cast<unsigned int>(dimX * dDimScalar) + newCx] = static_cast<unsigned short>(frame1[cy * dimX + cx]) << 8;
			atomicAdd(&hitCount[newCy * dimX + newCx], ones[cy * dimX + cx]);
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(2 * cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>((cx / 2) * 2 * fResolutionDivider)]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(dimY * fResolutionDivider * dimX * fResolutionDivider) + static_cast<unsigned int>(2 * cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>((cx / 2) * 2 * fResolutionDivider)]) * frameScalar / 2.0);

		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY / 2) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), static_cast<float>(dimX - 1));
			const int newCy = fminf(fmaxf(cy + offsetY, 0), (static_cast<float>(dimY) / 2) - 1);

			// U Channel
			if (cx % 2 == 0) {
				warpedFrame[static_cast<unsigned int>(dimY * dimX * dDimScalar) + newCy * static_cast<unsigned int>(dimX * dDimScalar) + (newCx / 2) * 2] = static_cast<unsigned short>(frame1[dimY * dimX + cy * dimX + (cx / 2) * 2]) << 8;

			// V Channel
			} else {
				warpedFrame[static_cast<unsigned int>(dimY * dimX * dDimScalar) + newCy * static_cast<unsigned int>(dimX * dDimScalar) + (newCx / 2) * 2 + 1] = static_cast<unsigned short>(frame1[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1]) << 8;
			}
		}
	}
}

// Kernel that warps a frame according to the offset array
__global__ void warpFrameKernelForBlending(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones,
										   unsigned char* warpedFrame, const float frameScalar, const int dimY, const int dimX, 
										   const float fResolutionDivider) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>(cx * fResolutionDivider)]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(dimY * fResolutionDivider * dimX * fResolutionDivider) + static_cast<unsigned int>(cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>(cx * fResolutionDivider)]) * frameScalar);

		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			warpedFrame[newCy * dimX + newCx] = frame1[cy * dimX + cx];
			atomicAdd(&hitCount[newCy * dimX + newCx], ones[cy * dimX + cx]);
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(2 * cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>((cx / 2) * 2 * fResolutionDivider)]) * frameScalar);
		const int offsetY = static_cast<int>((static_cast<float>(offsetArray[static_cast<unsigned int>(dimY * fResolutionDivider * dimX * fResolutionDivider) + static_cast<unsigned int>(2 * cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>((cx / 2) * 2 * fResolutionDivider)]) * frameScalar / 2.0));

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
__global__ void artifactRemovalKernelForOutput(const unsigned char* frame1, const int* hitCount, unsigned short* warpedFrame,
											   const unsigned int dimY, const unsigned int dimX, const double dDimScalar) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		if (hitCount[cy * dimX + cx] != 1) {
			warpedFrame[cy * static_cast<unsigned int>(dimX * dDimScalar) + cx] = static_cast<unsigned short>(frame1[cy * dimX + cx]) << 8;
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		if (hitCount[cy * dimX + cx] != 1) {
			// U Channel
			if (cx % 2 == 0) {
				warpedFrame[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + (cx / 2) * 2] = static_cast<unsigned short>(frame1[dimY * dimX + cy * dimX + (cx / 2) * 2]) << 8;

			// V Channel
			} else {
				warpedFrame[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + (cx / 2) * 2 + 1] = static_cast<unsigned short>(frame1[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1]) << 8;
			}
		}
	}
}

// Kernel that removes artifacts from the warped frame
__global__ void artifactRemovalKernelForBlending(const unsigned char* frame1, const int* hitCount, unsigned char* warpedFrame,
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
__global__ void blendFrameKernel(const unsigned char* frame1, const unsigned char* frame2, unsigned short* blendedFrame,
                                 const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                 const unsigned int dimX, const double dDimScalar) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		blendedFrame[cy * static_cast<unsigned int>(dimX * dDimScalar) + cx] = static_cast<unsigned short>(static_cast<float>(frame1[cy * dimX + cx]) *
					frame1Scalar + static_cast<float>(frame2[cy * dimX + cx]) * frame2Scalar) << 8;
	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		// U Channel
		if (cx % 2 == 0) {
			blendedFrame[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + (cx / 2) * 2] = static_cast<unsigned short>(static_cast<float>(frame1[dimY * dimX + cy * dimX + (cx / 2) * 2]) *
				frame1Scalar + static_cast<float>(frame2[dimY * dimX + cy * dimX + (cx / 2) * 2]) * frame2Scalar) << 8;

		// V Channel
		} else {
			blendedFrame[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + (cx / 2) * 2 + 1] = static_cast<unsigned short>(static_cast<float>(frame1[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1]) *
				frame1Scalar + static_cast<float>(frame2[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1]) * frame2Scalar) << 8;
		}
	}
}

// Kernel that creates an HSV flow image from the offset array
__global__ void convertFlowToHSVKernel(const int* flowArray, unsigned short* p010Array, const unsigned char* frame1,
                                       const unsigned int dimY, const unsigned int dimX, const float saturation,
                                       const float value, const float fResolutionDivider, const double dDimScalar) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Blend scalar
	constexpr float scalar = 0.5;

	// Get the current flow values
	const double x = flowArray[static_cast<unsigned int>(cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>(cx * fResolutionDivider)];
	const double y = flowArray[static_cast<unsigned int>(dimY * fResolutionDivider * dimX * fResolutionDivider) + static_cast<unsigned int>(cy * fResolutionDivider) * static_cast<unsigned int>(dimX * fResolutionDivider) + static_cast<unsigned int>(cx * fResolutionDivider)];

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
		default: rgb = { 0, 0, 0 }; break;
	}

	// Write the converted RGB values to the array
	if (cz < 2 && cy < dimY && cx < dimX) {
		// Y Channel
		if (cz == 0) {
			p010Array[cy * static_cast<unsigned int>(dimX * dDimScalar) + cx] = static_cast<unsigned short>((fmaxf(fminf(static_cast<float>(0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b), 255.0), 0.0)) * scalar + frame1[cy * dimX + cx] * (1.0 - scalar)) << 8;
		// U Channel
		} else if (cz == 1 && cx % 2 == 0 && cy < dimY / 2) {
			p010Array[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + (cx / 2) * 2] = static_cast<unsigned short>((fmaxf(fminf(static_cast<float>(0.492 * (rgb.b - (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b)) + 128), 255.0), 0.0) * scalar + frame1[cy * 2 * dimX + (cx / 2) * 2] * (1.0 - scalar))) << 8;
		// V Channel
		} else if (cy < dimY / 2) {
			p010Array[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + (cx / 2) * 2 + 1] = static_cast<unsigned short>((fmaxf(fminf(static_cast<float>(0.877 * (rgb.r - (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b)) + 128), 255.0), 0.0) * scalar + frame1[cy * 2 * dimX + (cx / 2) * 2 + 1] * (1.0 - scalar))) << 8;
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
* @param fResolutionScalar: The scalar to scale the resolution with
*/
void OpticalFlowCalc::init(const unsigned int dimY, const unsigned int dimX, const double dDimScalar, const float fResolutionScalar) {
	m_bIsInitialized = true;
	m_fResolutionScalar = fResolutionScalar;
	m_fResolutionDivider = 1.0f / fResolutionScalar;
	m_iDimX = dimX;
	m_iDimY = dimY;
	m_iLowDimX = static_cast<unsigned int>(static_cast<float>(dimX) * m_fResolutionDivider);
	m_iLowDimY = static_cast<unsigned int>(static_cast<float>(dimY) * m_fResolutionDivider);
	m_iLowDimZ = 5;
	m_dDimScalar = dDimScalar;
	m_lowGrid.x = static_cast<int>(fmaxf(ceilf(static_cast<float>(m_iLowDimX) / static_cast<float>(NUM_THREADS)), 1));
	m_lowGrid.y = static_cast<int>(fmaxf(ceilf(static_cast<float>(m_iLowDimY) / static_cast<float>(NUM_THREADS)), 1));
	m_lowGrid.z = 1;
	m_threads5.x = NUM_THREADS;
	m_threads5.y = NUM_THREADS;
	m_threads5.z = 5;
	m_threads2.x = NUM_THREADS;
	m_threads2.y = NUM_THREADS;
	m_threads2.z = 2;
	m_threads1.x = NUM_THREADS;
	m_threads1.y = NUM_THREADS;
	m_threads1.z = 1;
	m_grid.x = static_cast<int>(fmaxf(ceilf(dimX / static_cast<float>(NUM_THREADS)), 1));
	m_grid.y = static_cast<int>(fmaxf(ceilf(dimY / static_cast<float>(NUM_THREADS)), 1));
	m_frame1.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_frame2.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_imageDeltaArray.init({5, m_iLowDimY, m_iLowDimX});
	m_offsetArray12.init({2, 5, m_iLowDimY, m_iLowDimX});
	m_offsetArray21.init({2, m_iLowDimY, m_iLowDimX});
	m_blurredOffsetArray12.init({2, m_iLowDimY, m_iLowDimX});
	m_blurredOffsetArray21.init({2, m_iLowDimY, m_iLowDimX});
	m_statusArray.init({m_iLowDimY, m_iLowDimX});
	m_summedUpDeltaArray.init({5, m_iLowDimY, m_iLowDimX});
	m_normalizedDeltaArray.init({5, m_iLowDimY, m_iLowDimX});
	m_lowestLayerArray.init({m_iLowDimY, m_iLowDimX});
	m_warpedFrame12.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_warpedFrame21.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_outputFrame.init({1, dimY, dimX}, 0, static_cast<size_t>(3.0 * dimY * dimX * dDimScalar));
	m_hitCount12.init({1, dimY, dimX});
	m_hitCount21.init({1, dimY, dimX});
	m_ones.init({1, dimY, dimX}, 1);
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
* Converts a frame from NV12 to P010 format (stored in the output frame)
*
* @param p010Array: Pointer to the NV12 frame
*/
void OpticalFlowCalc::convertNV12toP010(const GPUArray<unsigned char>* nv12Array) {
	// Set the array entries to the provided value
	convertNV12toP010Kernel << <m_grid, m_threads2 >> > (nv12Array->arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Calculates the optical flow between frame1 and frame2
*
* @param iNumIterations: Number of iterations to calculate the optical flow
* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
*/
void OpticalFlowCalc::calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps) {
	// Reset variables
	unsigned int windowDimX = m_iLowDimX;
	unsigned int windowDimY = m_iLowDimY;
	if (iNumIterations == 0 || static_cast<float>(iNumIterations) > ceil(log2f(static_cast<float>(m_iLowDimX)))) {
		iNumIterations = static_cast<unsigned int>(ceil(log2f(static_cast<float>(m_iLowDimX))));
	}

	// Set the starting offset for the current window size
	setInitialOffset << <m_lowGrid, m_threads5 >> > (m_offsetArray12.arrayPtrGPU, m_iLowDimZ, m_iLowDimY, m_iLowDimX);

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		// Each step we adjust the offset array to find the ideal offset
		for (unsigned int step = 0; step < iNumSteps; step++) {
			// Reset the summed up delta array
			m_summedUpDeltaArray.zero();

			// 1. Calculate the image deltas with the current offset array
			if (m_bBisNewest) {
				calcImageDelta << <m_lowGrid, m_threads5 >> > (m_frame1.arrayPtrGPU, m_frame2.arrayPtrGPU,
															   m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
															   m_iLowDimZ, m_iLowDimY, m_iLowDimX, m_fResolutionScalar);
			} else {
				calcImageDelta << <m_lowGrid, m_threads5 >> > (m_frame2.arrayPtrGPU, m_frame1.arrayPtrGPU,
															   m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
															   m_iLowDimZ, m_iLowDimY, m_iLowDimX, m_fResolutionScalar);
			}

			// 2. Sum up the deltas of each window
			calcDeltaSums << <m_lowGrid, m_threads5 >> > (m_imageDeltaArray.arrayPtrGPU, m_summedUpDeltaArray.arrayPtrGPU,
														  windowDimY, windowDimX, m_iLowDimZ, m_iLowDimY, m_iLowDimX);

			// 3. Normalize the summed up delta array and find the best layer
			normalizeDeltaSums << <m_lowGrid, m_threads5 >> > (m_summedUpDeltaArray.arrayPtrGPU, m_lowestLayerArray.arrayPtrGPU,
															   m_offsetArray12.arrayPtrGPU, windowDimY, windowDimX,
															   m_iLowDimZ, m_iLowDimY, m_iLowDimX);

			// 4. Adjust the offset array based on the comparison results
			adjustOffsetArray << <m_lowGrid, m_threads1 >> > (m_offsetArray12.arrayPtrGPU, m_lowestLayerArray.arrayPtrGPU, 
															  m_statusArray.arrayPtrGPU, windowDimY, windowDimX, 
															  m_iLowDimZ, m_iLowDimY, m_iLowDimX);
		}

		// 5. Adjust window size
		windowDimX = max(windowDimX / 2, 1);
		windowDimY = max(windowDimY / 2, 1);

		// Reset the status array
		m_statusArray.zero();
	}

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
	m_offsetArray21.zero();

	// Launch kernel
	flipFlowKernel << <m_lowGrid, m_threads2 >> > (m_offsetArray12.arrayPtrGPU, m_offsetArray21.arrayPtrGPU,
												   m_iLowDimZ, m_iLowDimY, m_iLowDimX, m_fResolutionDivider);

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
	m_blurredOffsetArray12.zero();
	m_blurredOffsetArray21.zero();

	// Create CUDA streams
	cudaStream_t blurStream1, blurStream2;
	cudaStreamCreate(&blurStream1);
	cudaStreamCreate(&blurStream2);

	// Launch kernels
	blurKernel << <m_lowGrid, m_threads2, 0, blurStream1 >> > (m_offsetArray12.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU, kernelSize, m_iLowDimZ, m_iLowDimY, m_iLowDimX, true);
	blurKernel << <m_lowGrid, m_threads2, 0, blurStream2 >> > (m_offsetArray21.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU, kernelSize, 1, m_iLowDimY, m_iLowDimX, false);

	// Synchronize streams to ensure completion
	cudaStreamSynchronize(blurStream1);
	cudaStreamSynchronize(blurStream2);

	// Clean up streams
	cudaStreamDestroy(blurStream1);
	cudaStreamDestroy(blurStream2);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Warps the frames according to the calculated optical flow
*
* @param fScalar: The scalar to blend the frames with
* @param bOutput12: Whether to output the warped frame 12 or 21
*/
void OpticalFlowCalc::warpFramesForOutput(float fScalar, const bool bOutput12) {
	// Calculate the blend scalar
	const float frameScalar12 = fScalar;
	const float frameScalar21 = static_cast<float>(1.0) - fScalar;

	// Reset the hit count array
	if (bOutput12) {
		m_hitCount12.zero();
	} else {
		m_hitCount21.zero();
	}

	// Launch kernels
	if (m_bBisNewest) {
		// Frame 1 to Frame 2
		if (bOutput12) {
			warpFrameKernelForOutput << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
														          m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU, 
																  m_outputFrame.arrayPtrGPU, frameScalar12, m_iDimY, 
																  m_iDimX, m_fResolutionDivider, m_dDimScalar);
			artifactRemovalKernelForOutput << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount12.arrayPtrGPU, 
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		// Frame 2 to Frame 1
		} else {
			warpFrameKernelForOutput << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU, 
																  m_outputFrame.arrayPtrGPU, frameScalar21, m_iDimY, 
																  m_iDimX, m_fResolutionDivider, m_dDimScalar);
			artifactRemovalKernelForOutput << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount21.arrayPtrGPU, 
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		}
	} else {
		// Frame 1 to Frame 2
		if (bOutput12) {
			warpFrameKernelForOutput << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																  m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_outputFrame.arrayPtrGPU, frameScalar12, m_iDimY,
																  m_iDimX, m_fResolutionDivider, m_dDimScalar);
			artifactRemovalKernelForOutput << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																		m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		// Frame 2 to Frame 1
		} else {
			warpFrameKernelForOutput << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_outputFrame.arrayPtrGPU, frameScalar21, m_iDimY,
																  m_iDimX, m_fResolutionDivider, m_dDimScalar);
			artifactRemovalKernelForOutput << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		}
	}

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}

/*
* Warps the frames according to the calculated optical flow
*
* @param fScalar: The scalar to blend the frames with
*/
void OpticalFlowCalc::warpFramesForBlending(float fScalar) {
	// Calculate the blend scalar
	const float frameScalar12 = fScalar;
	const float frameScalar21 = static_cast<float>(1.0) - fScalar;

	// Reset the hit count array
	m_hitCount12.zero();
	m_hitCount21.zero();

	// Create CUDA streams
	cudaStream_t warpStream1, warpStream2;
	cudaStreamCreate(&warpStream1);
	cudaStreamCreate(&warpStream2);

	// Launch kernels
	if (m_bBisNewest) {
		// Frame 1 to Frame 2
		warpFrameKernelForBlending << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																			    m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU, 
																			    m_warpedFrame12.arrayPtrGPU, frameScalar12, m_iDimY, 
																			    m_iDimX, m_fResolutionDivider);
		artifactRemovalKernelForBlending << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_hitCount12.arrayPtrGPU, 
																				      m_warpedFrame12.arrayPtrGPU, m_iDimY, m_iDimX);

		// Frame 2 to Frame 1
		warpFrameKernelForBlending << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																			    m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU, 
																				m_warpedFrame21.arrayPtrGPU, frameScalar21, m_iDimY, 
																				m_iDimX, m_fResolutionDivider);
		artifactRemovalKernelForBlending << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_hitCount21.arrayPtrGPU, 
																					  m_warpedFrame21.arrayPtrGPU, m_iDimY, m_iDimX);

	} else {
		// Frame 1 to Frame 2
		warpFrameKernelForBlending << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																				m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU,
																				m_warpedFrame12.arrayPtrGPU, frameScalar12, m_iDimY,
																				m_iDimX, m_fResolutionDivider);
		artifactRemovalKernelForBlending << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame2.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																					  m_warpedFrame12.arrayPtrGPU, m_iDimY, m_iDimX);

		// Frame 2 to Frame 1
		warpFrameKernelForBlending << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																				m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU,
																				m_warpedFrame21.arrayPtrGPU, frameScalar21, m_iDimY,
																				m_iDimX, m_fResolutionDivider);
		artifactRemovalKernelForBlending << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame1.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																					  m_warpedFrame21.arrayPtrGPU, m_iDimY, m_iDimX);

	}

	// Synchronize streams to ensure completion
	cudaStreamSynchronize(warpStream1);
	cudaStreamSynchronize(warpStream2);

	// Clean up streams
	cudaStreamDestroy(warpStream1);
	cudaStreamDestroy(warpStream2);

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
void OpticalFlowCalc::blendFrames(float fScalar) {
	// Calculate the blend scalar
	const float frame1Scalar = static_cast<float>(1.0) - fScalar;
	const float frame2Scalar = fScalar;

	// Blend the frames
	blendFrameKernel << <m_grid, m_threads2 >> >(m_warpedFrame12.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
												 m_outputFrame.arrayPtrGPU, frame1Scalar, frame2Scalar,
	                                             m_iDimY, m_iDimX, m_dDimScalar);

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
void OpticalFlowCalc::drawFlowAsHSV(const float saturation, const float value) const {
	if (m_bBisNewest) {
		convertFlowToHSVKernel << <m_grid, m_threads2 >> > (m_blurredOffsetArray12.arrayPtrGPU, m_outputFrame.arrayPtrGPU,
														    m_frame2.arrayPtrGPU, m_iDimY, m_iDimX, 
															saturation, value, m_fResolutionDivider, m_dDimScalar);
	} else {
		convertFlowToHSVKernel << <m_grid, m_threads2 >> > (m_blurredOffsetArray12.arrayPtrGPU, m_outputFrame.arrayPtrGPU,
														    m_frame1.arrayPtrGPU, m_iDimY, m_iDimX, 
															saturation, value, m_fResolutionDivider, m_dDimScalar);
	}

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}