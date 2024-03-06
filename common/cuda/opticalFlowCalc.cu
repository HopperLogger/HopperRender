#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "opticalFlowCalc.cuh"

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
		int currX;
		int currY;

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
				
			// Adjust offset for next iteration
			case 6:
				statusArray[cy * dimX + cx] = 7;
				currX = offsetArray[cy * dimX + cx];
				currY = offsetArray[dimZ * dimY * dimX + cy * dimX + cx];

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

/*
* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
*/
void OpticalFlowCalc::flipFlow() const {
	// Reset the offset array
	m_offsetArray21.zero();

	// Launch kernel
	flipFlowKernel << <m_lowGrid, m_threads2 >> > (m_offsetArray12.arrayPtrGPU, m_offsetArray21.arrayPtrGPU,
												   m_iLowDimZ, m_iLowDimY, m_iLowDimX, m_dResolutionDivider);

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
void OpticalFlowCalc::blurFlowArrays(const int kernelSize) const {
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