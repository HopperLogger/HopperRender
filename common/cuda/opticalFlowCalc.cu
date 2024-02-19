#include <amvideo.h>
#include <chrono>
#include <iostream>
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

// Kernel that calculates the absolute difference between two frames using the offset array
__global__ void calcImageDelta(const unsigned char* frame1, const unsigned char* frame2, unsigned char* imageDeltaArray,
                               const int* offsetArray, const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = -static_cast<int>(static_cast<double>(offsetArray[cy * dimX + cx]));
		const int offsetY = -static_cast<int>(static_cast<double>(offsetArray[dimY * dimX + cy * dimX + cx]));

		// Current pixel is outside of frame
		if ((cy + offsetY < 0) || (cx + offsetX < 0) || (cy + offsetY > dimY) || (cx + offsetX > dimX)) {
			imageDeltaArray[cy * dimX + cx] = 0;
		// Current pixel is inside of frame
		} else {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			imageDeltaArray[cy * dimX + cx] = fabsf(frame1[newCy * dimX + newCx] - frame2[cy * dimX + cx]);
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		const int offsetX = -static_cast<int>(static_cast<double>(offsetArray[cy * 2 * dimX + (cx / 2) * 2]));
		const int offsetY = -static_cast<int>(static_cast<double>(offsetArray[dimY * dimX + cy * 2 * dimX + (cx / 2) * 2]) / 2.0);

		// Current pixel is outside of frame
		if ((cy + offsetY < 0) || (cx + offsetX < 0) || (cy + offsetY > dimY / 2) || (cx + offsetX > dimX)) {
			imageDeltaArray[dimY * dimX + cy * dimX + cx] = 128;
		// Current pixel is inside of frame
		} else {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), (dimY / 2) - 1);

			// U Channel
			if (cx % 2 == 0) {
				imageDeltaArray[dimY * dimX + cy * dimX + (cx / 2) * 2] = fabsf(frame1[dimY * dimX + newCy * dimX + (newCx / 2) * 2] - frame2[dimY * dimX + cy * dimX + (cx / 2) * 2]);

			// V Channel
			} else {
				imageDeltaArray[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1] = fabsf(frame1[dimY * dimX + newCy * dimX + (newCx / 2) * 2 + 1] - frame2[dimY * dimX + cy * dimX + (cx / 2) * 2 + 1]);
			}
		}
	}
}

// Kernel that sums up all the pixel deltas of each window
__global__ void calcDeltaSums(unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray,
                              const unsigned int windowDimY, const unsigned int windowDimX, const unsigned int dimY,
                              const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int windowIndexX = cx / windowDimX;
	const unsigned int windowIndexY = cy / windowDimY;

	// Check if the thread is inside the frame
	if (cz < 3 && cy < dimY && cx < dimX) {
		atomicAdd(&summedUpDeltaArray[(windowIndexY * windowDimY) * dimX + (windowIndexX * windowDimX)],
		          imageDeltaArray[cz + (3 * cy * dimX) + (3 * cx)]);
	}
}

// Kernel that normalizes all the pixel deltas of each window
__global__ void normalizeDeltaSums(const unsigned int* summedUpDeltaArray, float* normalizedDeltaArray,
                                   const int* offsetArray, const unsigned int windowDimY, const unsigned int windowDimX,
                                   const unsigned int dimY, const unsigned int dimX) {
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
		normalizedDeltaArray[cy * dimX + cx] = static_cast<float>(summedUpDeltaArray[cy * dimX + cx]) / static_cast<
			float>(numPixels);
	}
}

// Kernel that compares two arrays to find the lowest values
__global__ void compareArrays(const float* normalizedDeltaArrayOld, const float* normalizedDeltaArrayNew,
                              bool* isValueDecreasedArray, const unsigned int windowDimY, const unsigned int windowDimX,
                              const unsigned int dimY, const unsigned int dimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the thread is a window represent
	if (cy % windowDimY == 0 && cx % windowDimX == 0) {
		// Compare the two values
		isValueDecreasedArray[cy * dimX + cx] = normalizedDeltaArrayNew[cy * dimX + cx] < normalizedDeltaArrayOld[cy *
			dimX + cx];
	}
}

// Kernel that adjusts the offset array based on the comparison results
__global__ void adjustOffsetArray(int* offsetArray, const bool* isValueDecreasedArray, int* statusArray,
	const int currentGlobalOffset, const unsigned int windowDimY,
	const unsigned int windowDimX, const unsigned int dimY, const unsigned int dimX) {
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
				
			default:
				// Search is complete
				break;
		}
	}
}

// Kernel that warps frame1 according to the offset array
__global__ void warpFrameKernel(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones,
                                unsigned char* warpedFrame, const double frameScalar, const int dimY, const int dimX) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = static_cast<int>(static_cast<double>(offsetArray[cy * dimX + cx]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<double>(offsetArray[dimY * dimX + cy * dimX + cx]) * frameScalar);

		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			warpedFrame[newCy * dimX + newCx] = frame1[cy * dimX + cx];
			atomicAdd(&hitCount[newCy * dimX + newCx], ones[cy * dimX + cx]);
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<double>(offsetArray[cy * 2 * dimX + (cx / 2) * 2]) * frameScalar);
		const int offsetY = static_cast<int>((static_cast<double>(offsetArray[dimY * dimX + cy * 2 * dimX + (cx / 2) * 2]) * frameScalar) / 2.0);

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
__global__ void convertFlowToHSVKernel(const int* flowArray, unsigned char* RGBArray, const unsigned int dimZ,
                                       const unsigned int dimY, const unsigned int dimX, const double saturation,
                                       const double value, const float threshold) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Get the current flow values
	const double x = flowArray[cy * dimX + cx];
	const double y = flowArray[dimY * dimX + cy * dimX + cx];

	// Check if the flow is below the threshold
	if (fabsf(x) < threshold && fabsf(y) < threshold) {
		RGBArray[cz + (3 * cy * dimX) + (3 * cx)] = 0;
		return;
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
__global__ void flipFlowKernel(const int* flowArray12, int* flowArray21, const unsigned int dimY,
                               const unsigned int dimX) {
	// Current entry to be computed by the thread
	const int cx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	const int cy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const int cz = static_cast<int>(threadIdx.z);

	// Get the current flow values
	const int x = flowArray12[cy * dimX + cx];
	const int y = flowArray12[dimY * dimX + cy * dimX + cx];

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
		if (cz < dimZ && cy < dimY && cx < dimX) {
			for (int y = start; y < end; y++) {
				for (int x = start; x < end; x++) {
					if ((cy + y) < dimY && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
						atomicAdd(&blurredFlowArray[cz * dimY * dimX + cy * dimX + cx], flowArray[cz * dimY * dimX + (cy + y) * dimX + cx + x]);
					}
				}
			}
			blurredFlowArray[cz * dimY * dimX + cy * dimX + cx] /= (end - start) * (end - start);
		}
	} else {
		if (cz < dimZ && cy < dimY && cx < dimX) {
			blurredFlowArray[cz * dimY * dimX + cy * dimX + cx] = flowArray[cz * dimY * dimX + cy * dimX + cx];
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
	grid.x = fmaxf(ceilf(dimX / static_cast<float>(NUM_THREADS)), 1);
	grid.y = fmaxf(ceilf(dimY / static_cast<float>(NUM_THREADS)), 1);
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
	m_frame1.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_frame2.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_imageDeltaArray.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_offsetArray12.init({2, dimY, dimX});
	m_offsetArray21.init({2, dimY, dimX});
	m_blurredOffsetArray12.init({2, dimY, dimX});
	m_blurredOffsetArray21.init({2, dimY, dimX});
	m_rgboffsetArray.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_statusArray.init({dimY, dimX});
	m_summedUpDeltaArray.init({dimY, dimX});
	m_normalizedDeltaArrayA.init({dimY, dimX});
	m_normalizedDeltaArrayB.init({dimY, dimX});
	m_isValueDecreasedArray.init({dimY, dimX});
	m_warpedFrame12.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_warpedFrame21.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_blendedFrame.init({1, dimY, dimX}, 0, 1.5 * dimY * dimX);
	m_outputFrame.init({1, dimY, dimX}, 0, 3 * dimY * dimX);
	m_hitCount.init({1, dimY, dimX});
	m_ones.init({1, dimY, dimX}, 1);
	m_iWindowDimX = dimX;
	m_iWindowDimY = dimY;
	m_iCurrentGlobalOffset = 1;
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
*/
void OpticalFlowCalc::calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps) {
	const auto start = std::chrono::high_resolution_clock::now();

	// Reset variables
	m_iWindowDimX = m_frame1.dimX;
	m_iWindowDimY = m_frame1.dimY;
	m_iCurrentGlobalOffset = 1;
	if (iNumIterations == 0 || iNumIterations > ceil(log2f(m_frame1.dimX))) {
		iNumIterations = ceil(log2f(m_frame1.dimX));
	}

	// Reset the arrays
	m_offsetArray12.fill(0);
	m_summedUpDeltaArray.fill(0);

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		// Each step we adjust the offset array to find the ideal offset
		for (unsigned int step = 0; step < iNumSteps; step++) {
			// 1. Calculate the image deltas with the current offset array
			if (m_bBisNewest) {
				calcImageDelta << <grid, threads2 >> >(m_frame1.arrayPtrGPU, m_frame2.arrayPtrGPU,
				                                       m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
				                                       m_frame1.dimY, m_frame1.dimX);
			} else {
				calcImageDelta << <grid, threads2 >> >(m_frame2.arrayPtrGPU, m_frame1.arrayPtrGPU,
				                                       m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
				                                       m_frame1.dimY, m_frame1.dimX);
			}
			// 2. Sum up the deltas of each window
			calcDeltaSums << <grid, threads2 >> >(m_imageDeltaArray.arrayPtrGPU, m_summedUpDeltaArray.arrayPtrGPU,
			                                      m_iWindowDimY, m_iWindowDimX, m_frame1.dimY, m_frame1.dimX);

			// Switch between the two normalized delta arrays to avoid copying
			if (step % 2 == 0) {
				// 3. Normalize the summed up delta array
				normalizeDeltaSums << <grid, threads1 >> >(m_summedUpDeltaArray.arrayPtrGPU,
				                                           m_normalizedDeltaArrayB.arrayPtrGPU,
				                                           m_offsetArray12.arrayPtrGPU, m_iWindowDimY, m_iWindowDimX,
				                                           m_frame1.dimY, m_frame1.dimX);

				// 4. Check if the new normalized delta array is better than the old one
				if (step > 0) {
					compareArrays << <grid, threads1 >> > (m_normalizedDeltaArrayA.arrayPtrGPU,
						m_normalizedDeltaArrayB.arrayPtrGPU,
						m_isValueDecreasedArray.arrayPtrGPU, m_iWindowDimY, m_iWindowDimX,
						m_frame1.dimY, m_frame1.dimX);
				}
			} else {
				// 3. Normalize the summed up delta array
				normalizeDeltaSums << <grid, threads1 >> >(m_summedUpDeltaArray.arrayPtrGPU,
				                                           m_normalizedDeltaArrayA.arrayPtrGPU,
				                                           m_offsetArray12.arrayPtrGPU, m_iWindowDimY, m_iWindowDimX,
				                                           m_frame1.dimY, m_frame1.dimX);

				// 4. Check if the new normalized delta array is better than the old one
				compareArrays << <grid, threads1 >> >(m_normalizedDeltaArrayB.arrayPtrGPU,
				                                      m_normalizedDeltaArrayA.arrayPtrGPU,
				                                      m_isValueDecreasedArray.arrayPtrGPU, m_iWindowDimY, m_iWindowDimX,
				                                      m_frame1.dimY, m_frame1.dimX);
			}

			// 5. Adjust the offset array based on the comparison results
			adjustOffsetArray << <grid, threads1 >> >(m_offsetArray12.arrayPtrGPU,
			                                          m_isValueDecreasedArray.arrayPtrGPU, m_statusArray.arrayPtrGPU,
			                                          m_iCurrentGlobalOffset, m_iWindowDimY, m_iWindowDimX,
				                                      m_frame1.dimY, m_frame1.dimX);

			// Wait for all threads to finish
			cudaDeviceSynchronize();

			// Reset the summed up delta array
			m_summedUpDeltaArray.fill(0);
		}
		// 6. Adjust window size
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

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration<double, std::milli>(stop - start).count();
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
	m_hitCount.fill(0);

	// Warp the frame
	if (m_bBisNewest) {
		warpFrameKernel << <grid, threads2 >> >(m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
		                                        m_hitCount.arrayPtrGPU, m_ones.arrayPtrGPU, m_warpedFrame12.arrayPtrGPU,
		                                        frameScalar, m_frame1.dimY, m_frame1.dimX);
		artifactRemovalKernel << <grid, threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount.arrayPtrGPU, m_warpedFrame12.arrayPtrGPU, m_frame1.dimY, m_frame1.dimX);
	} else {
		warpFrameKernel << <grid, threads2 >> >(m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
		                                        m_hitCount.arrayPtrGPU, m_ones.arrayPtrGPU, m_warpedFrame12.arrayPtrGPU,
		                                        frameScalar, m_frame1.dimY, m_frame1.dimX);
		artifactRemovalKernel << <grid, threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount.arrayPtrGPU, m_warpedFrame12.arrayPtrGPU, m_frame1.dimY, m_frame1.dimX);
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
	m_hitCount.fill(0);

	// Warp the frame
	if (m_bBisNewest) {
		warpFrameKernel << <grid, threads2 >> >(m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
		                                        m_hitCount.arrayPtrGPU, m_ones.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
		                                        frameScalar, m_frame1.dimY, m_frame1.dimX);
		artifactRemovalKernel << <grid, threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU, m_frame2.dimY, m_frame2.dimX);
	} else {
		warpFrameKernel << <grid, threads2 >> >(m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
		                                        m_hitCount.arrayPtrGPU, m_ones.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
		                                        frameScalar, m_frame1.dimY, m_frame1.dimX);
		artifactRemovalKernel << <grid, threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU, m_frame2.dimY, m_frame2.dimX);
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
	blendFrameKernel << <grid, threads2 >> >(m_warpedFrame12.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
	                                         m_blendedFrame.arrayPtrGPU, frame1Scalar, frame2Scalar,
	                                         m_warpedFrame12.dimY, m_warpedFrame12.dimX);

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
void OpticalFlowCalc::downloadFlowAsHSV(unsigned char* memPointer, const double saturation, const double value,
                                        const float threshold) const {
	// Launch kernel
	convertFlowToHSVKernel << <grid, threads3 >> >(m_blurredOffsetArray12.arrayPtrGPU, m_rgboffsetArray.arrayPtrGPU, 3,
	                                               m_offsetArray12.dimY, m_offsetArray12.dimX, saturation, value,
	                                               threshold);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Copy host array to GPU
	cudaMemcpy(memPointer, m_rgboffsetArray.arrayPtrGPU, 3 * m_offsetArray12.dimY * m_offsetArray12.dimX,
	           cudaMemcpyDeviceToHost);
}

/*
* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
*/
void OpticalFlowCalc::flipFlow() {
	// Reset the offset array
	m_offsetArray21.fill(0);

	// Launch kernel
	flipFlowKernel << <grid, threads2 >> >(m_offsetArray12.arrayPtrGPU, m_offsetArray21.arrayPtrGPU,
	                                       m_offsetArray12.dimY, m_offsetArray12.dimX);

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
	blurKernel << <grid, threads2 >> > (m_offsetArray12.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU, kernelSize, 2, m_blurredOffsetArray12.dimY, m_blurredOffsetArray12.dimX);
	blurKernel << <grid, threads2 >> > (m_offsetArray21.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU, kernelSize, 2, m_blurredOffsetArray21.dimY, m_blurredOffsetArray21.dimX);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}