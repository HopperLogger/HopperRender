#include <amvideo.h>
#include <iomanip>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "opticalFlowCalcSDR.cuh"

// Debug message function
void CudaDebugMessage(const std::string& message, const bool showLog) {
	if (showLog) {
		const std::string m_debugMessage = message + "\n";
		OutputDebugStringA(m_debugMessage.c_str());
	}
}

// Kernel that converts an NV12 array to a P010 array
__global__ void convertNV12toP010KernelSDR(const unsigned char* nv12Array, unsigned short* p010Array, const unsigned int dimY,
										   const unsigned int dimX, const int scaledDimY, const int scaledDimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Check if the current thread is inside the Y-Channel or the U/V-Channel
	if ((cz == 0 && cy < scaledDimY && cx < scaledDimX) || (cz == 1 && cy < (scaledDimY / 2) && cx < scaledDimX)) {
		p010Array[cz * dimY * scaledDimX + cy * scaledDimX + cx] = static_cast<unsigned short>(nv12Array[cz * dimY * dimX + cy * dimX + cx]) << 8;
	}
}

// Kernel that blurs a frame
__global__ void blurFrameKernelSDR(const unsigned char* frameArray, unsigned char* blurredFrameArray, 
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

// Kernel that warps a frame according to the offset array
__global__ void warpFrameKernelForOutputSDR(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones,
									        unsigned short* warpedFrame, const float frameScalar, const int lowDimY, const int lowDimX,
											const int dimY, const int dimX, const double resolutionDivider,
											const int directionIdxOffset, const int scaledDimX, const int channelIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	const int scaledCx = static_cast<int>(cx * resolutionDivider); // The X-Index of the current thread in the offset array
	const int scaledCy = static_cast<int>(cy * resolutionDivider); // The Y-Index of the current thread in the offset array

	// Y-Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[scaledCy * lowDimX + scaledCx]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + scaledCy * lowDimX + scaledCx]) * frameScalar);
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < dimY && newCx >= 0 && newCx < dimX) {
			warpedFrame[newCy * scaledDimX + newCx] = static_cast<unsigned short>(frame1[cy * dimX + cx]) << 8;
			atomicAdd(&hitCount[newCy * dimX + newCx], ones[cy * dimX + cx]);
		}

	// U/V-Channel
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[2 * scaledCy * lowDimX + (scaledCx / 2) * 2]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + 2 * scaledCy * lowDimX + (scaledCx / 2) * 2]) * frameScalar / 2.0);
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < dimY / 2 && newCx >= 0 && newCx < dimX) {
			// U-Channel
			if ((cx & 1) == 0) {
				warpedFrame[channelIdxOffset + newCy * scaledDimX + (newCx / 2) * 2] = static_cast<unsigned short>(frame1[dimY * dimX + cy * dimX + cx]) << 8;

			// V-Channel
			} else {
				warpedFrame[channelIdxOffset + newCy * scaledDimX + (newCx / 2) * 2 + 1] = static_cast<unsigned short>(frame1[dimY * dimX + cy * dimX + cx]) << 8;
			}
		}
	}
}

// Kernel that warps a frame according to the offset array
__global__ void warpFrameKernelForBlendingSDR(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones,
										      unsigned char* warpedFrame, const float frameScalar, const int lowDimY, const int lowDimX,
											  const int dimY, const int dimX, const double resolutionDivider,
											  const int directionIdxOffset, const int channelIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	const int scaledCx = static_cast<int>(cx * resolutionDivider); // The X-Index of the current thread in the offset array
	const int scaledCy = static_cast<int>(cy * resolutionDivider); // The Y-Index of the current thread in the offset array

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[scaledCy * lowDimX + scaledCx]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + scaledCy * lowDimX + scaledCx]) * frameScalar);
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < dimY && newCx >= 0 && newCx < dimX) {
			warpedFrame[newCy * dimX + newCx] = frame1[cy * dimX + cx];
			atomicAdd(&hitCount[newCy * dimX + newCx], ones[cy * dimX + cx]);
		}

	// U/V-Channel
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[2 * scaledCy * lowDimX + (scaledCx / 2) * 2]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + 2 * scaledCy * lowDimX + (scaledCx / 2) * 2]) * frameScalar / 2.0);
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < dimY / 2 && newCx >= 0 && newCx < dimX) {
			// U-Channel
			if ((cx & 1) == 0) {
				warpedFrame[channelIdxOffset + newCy * dimX + (newCx / 2) * 2] = frame1[dimY * dimX + cy * dimX + cx];

			// V-Channel
			} else {
				warpedFrame[channelIdxOffset + newCy * dimX + (newCx / 2) * 2 + 1] = frame1[dimY * dimX + cy * dimX + cx];
			}
		}
	}
}

// Kernel that removes artifacts from the warped frame
__global__ void artifactRemovalKernelForOutputSDR(const unsigned char* frame1, const int* hitCount, unsigned short* warpedFrame,
										     	  const unsigned int dimY, const unsigned int dimX, const int scaledDimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int threadIndex2D = cy * dimX + cx; // Standard thread index without Z-Dim

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[cy * scaledDimX + cx] = static_cast<unsigned short>(frame1[threadIndex2D]) << 8;
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[dimY * scaledDimX + cy * scaledDimX + cx] = static_cast<unsigned short>(frame1[dimY * dimX + threadIndex2D]) << 8;
		}
	}
}

// Kernel that blends frame1 to frame2
__global__ void blendFrameKernelSDR(const unsigned char* frame1, const unsigned char* frame2, unsigned short* blendedFrame,
                                    const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                    const unsigned int dimX, const int scaledDimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		blendedFrame[cy * scaledDimX + cx] = 
			static_cast<unsigned short>(
				static_cast<float>(frame1[cy * dimX + cx]) * frame1Scalar + 
				static_cast<float>(frame2[cy * dimX + cx]) * frame2Scalar
			) << 8;
	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		blendedFrame[dimY * scaledDimX + cy * scaledDimX + cx] = 
			static_cast<unsigned short>(
				static_cast<float>(frame1[dimY * dimX + cy * dimX + cx]) * frame1Scalar + 
				static_cast<float>(frame2[dimY * dimX + cy * dimX + cx]) * frame2Scalar
			) << 8;
	}
}

// Kernel that creates an HSV flow image from the offset array
__global__ void convertFlowToHSVKernelSDR(const int* flowArray, unsigned short* p010Array, const unsigned char* frame1,
                                          const double blendScalar, const int lowDimX, const unsigned int dimY, const unsigned int dimX, 
										  const double resolutionDivider, const int directionIdxOffset, const int scaledDimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	const unsigned int scaledCx = static_cast<unsigned int>(cx * resolutionDivider); // The X-Index of the current thread in the offset array
	const unsigned int scaledCy = static_cast<unsigned int>(cy * resolutionDivider); // The Y-Index of the current thread in the offset array

	// Get the current flow values
	double x;
	double y;
	if (cz == 0) {
		x = flowArray[scaledCy * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + scaledCy * lowDimX + scaledCx];
	} else {
		x = flowArray[scaledCy * 2 * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + scaledCy * 2 * lowDimX + scaledCx];
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
	const int h_i = static_cast<int>(hue * 6.0);
	const double f = hue * 6.0 - h_i;
	const double q = 1.0 - f;

	RGB rgb;
	switch (h_i % 6) {
		case 0: rgb = { static_cast<int>(255), static_cast<int>(f * 255), 0 }; break;
		case 1: rgb = { static_cast<int>(q * 255), static_cast<int>(255), 0 }; break;
		case 2: rgb = { 0, static_cast<int>(255), static_cast<int>(f * 255) }; break;
		case 3: rgb = { 0, static_cast<int>(q * 255), static_cast<int>(255) }; break;
		case 4: rgb = { static_cast<int>(f * 255), 0, static_cast<int>(255) }; break;
		case 5: rgb = { static_cast<int>(255), 0, static_cast<int>(q * 255) }; break;
		default: rgb = { 0, 0, 0 }; break;
	}

	// Prevent random colors when there is no flow
	if (fabs(x) < 1.0 && fabs(y) < 1.0) {
		rgb = { 0, 0, 0 };
	}

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		p010Array[cy * scaledDimX + cx] = 
			static_cast<unsigned short>(
				(fmax(fmin(0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b, 255.0), 0.0)) * blendScalar + 
				frame1[cy * dimX + cx] * (1.0 - blendScalar)
			) << 8;
	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		// U Channel
		if ((cx & 1) == 0) {
			p010Array[dimY * scaledDimX + cy * scaledDimX + (cx / 2) * 2] = 
				static_cast<unsigned short>(
					fmax(fmin(0.492 * (rgb.b - (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b)) + 128, 255.0), 0.0)
				) << 8;
		// V Channel
		} else {
			p010Array[dimY * scaledDimX + cy * scaledDimX + (cx / 2) * 2 + 1] = 
				static_cast<unsigned short>(
					fmax(fmin(0.877 * (rgb.r - (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b)) + 128, 255.0), 0.0)
				) << 8;
		}
	}
}

/*
* Initializes the SDR optical flow calculator
*
* @param dimY: The height of the frame
* @param dimX: The width of the frame
* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
* @param dResolutionDivider: The scalar to scale the resolution with
*/
OpticalFlowCalcSDR::OpticalFlowCalcSDR(const unsigned int dimY, const unsigned int dimX, const double dDimScalar, const double dResolutionDivider) {
	m_dResolutionDivider = dResolutionDivider;
	m_dResolutionScalar = 1.0 / dResolutionDivider;
	m_iDimX = dimX;
	m_iDimY = dimY;
	m_iLowDimX = static_cast<unsigned int>(static_cast<double>(dimX) * m_dResolutionDivider);
	m_iLowDimY = static_cast<unsigned int>(static_cast<double>(dimY) * m_dResolutionDivider);
	m_iNumLayers = 5;
	m_dDimScalar = dDimScalar;
	m_lowGrid.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / static_cast<double>(NUM_THREADS)), 1.0));
	m_lowGrid.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / static_cast<double>(NUM_THREADS)), 1.0));
	m_lowGrid.z = 1;
	m_gridCID.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / static_cast<double>(16)), 1.0));
	m_gridCID.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / static_cast<double>(16)), 1.0));
	m_gridCID.z = m_iNumLayers;
	m_gridAOA.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / static_cast<double>(32)), 1.0));
	m_gridAOA.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / static_cast<double>(32)), 1.0));
	m_gridAOA.z = 1;
	m_threadsCID.x = 16;
	m_threadsCID.y = 16;
	m_threadsCID.z = 2;
	m_threadsAOA.x = 32;
	m_threadsAOA.y = 32;
	m_threadsAOA.z = 1;
	m_threads10.x = NUM_THREADS;
	m_threads10.y = NUM_THREADS;
	m_threads10.z = 10;
	m_threads5.x = NUM_THREADS;
	m_threads5.y = NUM_THREADS;
	m_threads5.z = 5;
	m_threads2.x = NUM_THREADS;
	m_threads2.y = NUM_THREADS;
	m_threads2.z = 2;
	m_threads1.x = NUM_THREADS;
	m_threads1.y = NUM_THREADS;
	m_threads1.z = 1;
	m_grid.x = static_cast<int>(fmax(ceil(dimX / static_cast<double>(NUM_THREADS)), 1.0));
	m_grid.y = static_cast<int>(fmax(ceil(dimY / static_cast<double>(NUM_THREADS)), 1.0));
	m_frame1.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_frame2.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_blurredFrame1.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_blurredFrame2.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_imageDeltaArray.init({5, 2, dimY, dimX});
	m_offsetArray12.init({2, 5, dimY, dimX});
	m_offsetArray21.init({2, dimY, dimX});
	m_blurredOffsetArray12.init({2, dimY, dimX});
	m_blurredOffsetArray21.init({2, dimY, dimX});
	m_statusArray.init({dimY, dimX});
	m_summedUpDeltaArray.init({5, dimY, dimX});
	m_normalizedDeltaArray.init({5, dimY, dimX});
	m_lowestLayerArray.init({dimY, dimX});
	m_warpedFrame12.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_warpedFrame21.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_outputFrame.init({1, dimY, dimX}, 0, static_cast<size_t>(3.0 * dimY * dimX * dDimScalar));
	m_hitCount12.init({1, dimY, dimX});
	m_hitCount21.init({1, dimY, dimX});
	m_ones.init({1, dimY, dimX}, 1);
}

/*
* Updates the frame1 array
*
* @param pInBuffer: Pointer to the input frame
*/
void OpticalFlowCalcSDR::updateFrame1(const unsigned char* pInBuffer) {
	m_frame1.fillData(pInBuffer);
	m_bBisNewest = false;
}

/*
* Updates the frame2 array
*
* @param pInBuffer: Pointer to the input frame
*/
void OpticalFlowCalcSDR::updateFrame2(const unsigned char* pInBuffer) {
	m_frame2.fillData(pInBuffer);
	m_bBisNewest = true;
}

/*
* Copies the frame in the correct format to the output buffer
*
* @param pInBuffer: Pointer to the input frame
* @param pOutBuffer: Pointer to the output frame
*/
void OpticalFlowCalcSDR::copyFrame(const unsigned char* pInBuffer, unsigned char* pOutBuffer) {
	// Set the array entries to the provided value
	m_frame1.fillData(pInBuffer);

	// Convert the NV12 frame to P010
	convertNV12toP010KernelSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iDimY * m_dDimScalar, m_iDimX * m_dDimScalar);

	// Download the output frame
	m_outputFrame.download(pOutBuffer);
}

/*
* Blurs a frame
*
* @param kernelSize: Size of the kernel to use for the blur
* @param directOutput: Whether to output the blurred frame directly
*/
void OpticalFlowCalcSDR::blurFrameArray(const unsigned char kernelSize, const bool directOutput) {
	const unsigned char boundsOffset = kernelSize >> 1;
	const unsigned char chacheSize = kernelSize + (boundsOffset << 1);
	const size_t sharedMemSize = chacheSize * chacheSize * sizeof(unsigned char);
	const unsigned short totalThreads = max(kernelSize * kernelSize, 1);
    const unsigned short totalEntries = chacheSize * chacheSize;
    const unsigned char avgEntriesPerThread = totalEntries / totalThreads;
	const unsigned short remainder = totalEntries % totalThreads;
	const char lumStart = -(kernelSize >> 1);
	const unsigned char lumEnd = (kernelSize >> 1);
	const char chromStart = -(kernelSize >> 2);
	const unsigned char chromEnd = (kernelSize >> 2);
	const unsigned short lumPixelCount = (lumEnd - lumStart) * (lumEnd - lumStart);
	const unsigned short chromPixelCount = (chromEnd - chromStart) * (chromEnd - chromStart);

	// Calculate the number of blocks needed
	const unsigned int NUM_BLOCKS_X = max(static_cast<int>(ceil(static_cast<float>(m_iDimX) / kernelSize)), 1);
	const unsigned int NUM_BLOCKS_Y = max(static_cast<int>(ceil(static_cast<float>(m_iDimY) / kernelSize)), 1);

	// Use dim3 structs for block and grid size
	dim3 gridBF(NUM_BLOCKS_X, NUM_BLOCKS_Y, 2);
	dim3 threadsBF(kernelSize, kernelSize, 1);

	if (!m_bBisNewest) {
		// No need to blur the frame if the kernel size is less than 4
		if (kernelSize < 4) {
			cudaMemcpy(m_blurredFrame1.arrayPtrGPU, m_frame1.arrayPtrGPU, m_frame1.bytes, cudaMemcpyDeviceToDevice);
		} else {
			// Launch kernel
			blurFrameKernelSDR << <gridBF, threadsBF, sharedMemSize >> > (m_frame1.arrayPtrGPU, m_blurredFrame1.arrayPtrGPU, kernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, lumStart, lumEnd, lumPixelCount, chromStart, chromEnd, chromPixelCount, m_iDimY, m_iDimX);
		}

		// Convert the NV12 frame to P010 if we are doing direct output
		if (directOutput) {
			convertNV12toP010KernelSDR << <m_grid, m_threads2 >> > (m_blurredFrame1.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iDimY * m_dDimScalar, m_iDimX * m_dDimScalar);
		}
	} else {
		// No need to blur the frame if the kernel size is less than 4
		if (kernelSize < 4) {
			cudaMemcpy(m_blurredFrame2.arrayPtrGPU, m_frame2.arrayPtrGPU, m_frame1.bytes, cudaMemcpyDeviceToDevice);
		} else {
			// Launch kernel
			blurFrameKernelSDR << <gridBF, threadsBF, sharedMemSize >> > (m_frame2.arrayPtrGPU, m_blurredFrame2.arrayPtrGPU, kernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, lumStart, lumEnd, lumPixelCount, chromStart, chromEnd, chromPixelCount, m_iDimY, m_iDimX);
		}

		// Convert the NV12 frame to P010 if we are doing direct output
		if (directOutput) {
			convertNV12toP010KernelSDR << <m_grid, m_threads2 >> > (m_blurredFrame2.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iDimY * m_dDimScalar, m_iDimX * m_dDimScalar);
		}
	}
}

/*
* Calculates the optical flow between frame1 and frame2
*
* @param iNumIterations: Number of iterations to calculate the optical flow
* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
*/
void OpticalFlowCalcSDR::calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps) {
	// Reset variables
	const unsigned int directionIdxOffset = m_iNumLayers * m_iLowDimY * m_iLowDimX; // Offset to index the Y-Offset-Layer
	const unsigned int channelIdxOffset = m_iDimY * m_iDimX; // Offset to index the color channel of the current thread
	const unsigned int layerIdxOffset = m_iLowDimY * m_iLowDimX; // Offset to index the layer of the current thread
	unsigned int iNumStepsPerIter = iNumSteps; // Number of steps executed to find the ideal offset (limits the maximum offset)

	// We set the initial window size to the next larger power of 2
	unsigned int windowDim = 1;
	unsigned int maxDim = max(m_iLowDimX, m_iLowDimY);
    if (maxDim && !(maxDim & (maxDim - 1))) {
		windowDim = maxDim;
	} else {
		while (maxDim & (maxDim - 1)) {
			maxDim &= (maxDim - 1);
		}
		windowDim = maxDim << 1;
	}

	if (iNumIterations == 0 || static_cast<double>(iNumIterations) > ceil(log2(windowDim))) {
		iNumIterations = static_cast<unsigned int>(ceil(log2(windowDim))) + 1;
	}

	unsigned int num_threads = min(windowDim, 16);
	size_t sharedMemSize = num_threads * num_threads * sizeof(unsigned int);

	// Calculate the number of blocks needed
	unsigned int NUM_BLOCKS_X = max(static_cast<int>(ceil(static_cast<float>(m_imageDeltaArray.dimX) / num_threads)), 1);
	unsigned int NUM_BLOCKS_Y = max(static_cast<int>(ceil(static_cast<float>(m_imageDeltaArray.dimY) / num_threads)), 1);

	// Use dim3 structs for block and grid size
	dim3 gridCDS(NUM_BLOCKS_X, NUM_BLOCKS_Y, 5);
	dim3 threadsCDS(num_threads, num_threads, 1);

	// Set the starting offset for the current window size
	setInitialOffset << <m_lowGrid, m_threads5 >> > (m_offsetArray12.arrayPtrGPU, m_iNumLayers, m_iLowDimY, m_iLowDimX);

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		switch (iter) {
			case 0: iNumStepsPerIter = iNumSteps; break;
			case 1: iNumStepsPerIter = iNumSteps; break;
			case 2: iNumStepsPerIter = iNumSteps; break;
			case 3: iNumStepsPerIter = iNumSteps / 2; break;
			case 4: iNumStepsPerIter = iNumSteps / 2; break;
			case 5: iNumStepsPerIter = iNumSteps / 2; break;
			case 6: iNumStepsPerIter = iNumSteps / 3; break;
			case 7: iNumStepsPerIter = iNumSteps / 3; break;
			case 8: iNumStepsPerIter = iNumSteps / 6; break;
			default: iNumStepsPerIter = 1; break;
		}
		// Each step we adjust the offset array to find the ideal offset
		for (unsigned int step = 0; step < iNumStepsPerIter; step++) {
			// Reset the summed up delta array
			m_summedUpDeltaArray.zero();

			// 1. Calculate the image deltas with the current offset array
			if (m_bBisNewest) {
				calcImageDelta << <m_gridCID, m_threadsCID >> > (m_blurredFrame1.arrayPtrGPU, m_blurredFrame2.arrayPtrGPU,
															   m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
															   m_iLowDimY, m_iLowDimX, m_iDimY, m_iDimX,
															   static_cast<float>(m_dResolutionScalar), directionIdxOffset, channelIdxOffset);
			} else {
				calcImageDelta << <m_gridCID, m_threadsCID >> > (m_blurredFrame2.arrayPtrGPU, m_blurredFrame1.arrayPtrGPU,
															   m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
															   m_iLowDimY, m_iLowDimX, m_iDimY, m_iDimX,
															   static_cast<float>(m_dResolutionScalar), directionIdxOffset, channelIdxOffset);
			}

			// 2. Sum up the deltas of each window
			if (windowDim >= 8) {
				calcDeltaSums8x8 << <gridCDS, threadsCDS, sharedMemSize>> > (m_imageDeltaArray.arrayPtrGPU, m_summedUpDeltaArray.arrayPtrGPU, 
																	2 * m_iLowDimY * m_iLowDimX, m_iLowDimY * m_iLowDimX, 
																	m_iLowDimY, m_iLowDimX, windowDim);
			} else if (windowDim == 4) {
				calcDeltaSums4x4 << <gridCDS, threadsCDS, sharedMemSize>> > (m_imageDeltaArray.arrayPtrGPU, m_summedUpDeltaArray.arrayPtrGPU, 
																	2 * m_iLowDimY * m_iLowDimX, m_iLowDimY * m_iLowDimX, 
																	m_iLowDimY, m_iLowDimX, windowDim);
			} else if (windowDim == 2) {
				calcDeltaSums2x2 << <gridCDS, threadsCDS, sharedMemSize>> > (m_imageDeltaArray.arrayPtrGPU, m_summedUpDeltaArray.arrayPtrGPU, 
																	2 * m_iLowDimY * m_iLowDimX, m_iLowDimY * m_iLowDimX, 
																	m_iLowDimY, m_iLowDimX, windowDim);
			} else if (windowDim == 1) {
				calcDeltaSums1x1 << <m_lowGrid, m_threads5, sharedMemSize>> > (m_imageDeltaArray.arrayPtrGPU, m_summedUpDeltaArray.arrayPtrGPU, 
																	2 * m_iLowDimY * m_iLowDimX, m_iLowDimY * m_iLowDimX, 
																	m_iLowDimY, m_iLowDimX, windowDim);
			}

			// 3. Normalize the summed up delta array and find the best layer
			normalizeDeltaSums << <m_lowGrid, m_threads5 >> > (m_summedUpDeltaArray.arrayPtrGPU, m_lowestLayerArray.arrayPtrGPU,
															   m_offsetArray12.arrayPtrGPU, windowDim, windowDim * windowDim,
															   directionIdxOffset, layerIdxOffset, m_iNumLayers, m_iLowDimY, m_iLowDimX);

			// 4. Adjust the offset array based on the comparison results
			adjustOffsetArray << <m_gridAOA, m_threadsAOA >> > (m_offsetArray12.arrayPtrGPU, m_lowestLayerArray.arrayPtrGPU,
															  m_statusArray.arrayPtrGPU, windowDim, directionIdxOffset, layerIdxOffset,
															  m_iNumLayers, m_iLowDimY, m_iLowDimX, step == iNumStepsPerIter - 1);
		}

		// 5. Adjust variables for the next iteration
		windowDim = max(windowDim >> 1, 1);
		num_threads = max(min(windowDim, 16), 8);
		sharedMemSize = num_threads * num_threads * sizeof(unsigned int);
		NUM_BLOCKS_X = max(static_cast<int>(ceil(static_cast<float>(m_imageDeltaArray.dimX) / num_threads)), 1);
		NUM_BLOCKS_Y = max(static_cast<int>(ceil(static_cast<float>(m_imageDeltaArray.dimY) / num_threads)), 1);
		gridCDS.x = NUM_BLOCKS_X;
		gridCDS.y = NUM_BLOCKS_Y;
		threadsCDS.x = num_threads;
		threadsCDS.y = num_threads;

		// Reset the status array
		m_statusArray.zero();
	}
}

/*
* Warps the frames according to the calculated optical flow
*
* @param fScalar: The scalar to blend the frames with
* @param bOutput12: Whether to output the warped frame 12 or 21
*/
void OpticalFlowCalcSDR::warpFramesForOutput(float fScalar, const bool bOutput12) {
	// Calculate the blend scalar
	const float frameScalar12 = fScalar;
	const float frameScalar21 = static_cast<float>(1.0) - fScalar;

	// Calculate variables so the threds don't have to do it
	const int directionIdxOffset = m_iDimY * m_dResolutionDivider * m_iDimX * m_dResolutionDivider;
	const int scaledDimX = static_cast<unsigned int>(m_iDimX * m_dDimScalar);
	const int channelIdxOffset = static_cast<unsigned int>(m_iDimY * m_iDimX * m_dDimScalar);

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
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
														          m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU, 
																  m_outputFrame.arrayPtrGPU, frameScalar12, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, scaledDimX, channelIdxOffset);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, scaledDimX);
		// Frame 2 to Frame 1
		} else {
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU, 
																  m_outputFrame.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, scaledDimX, channelIdxOffset);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, scaledDimX);
		}
	} else {
		// Frame 1 to Frame 2
		if (bOutput12) {
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																  m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_outputFrame.arrayPtrGPU, frameScalar12, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, scaledDimX, channelIdxOffset);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																		m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, scaledDimX);
		// Frame 2 to Frame 1
		} else {
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_outputFrame.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, scaledDimX, channelIdxOffset);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, scaledDimX);
		}
	}
}

/*
* Warps the frames according to the calculated optical flow
*
* @param fScalar: The scalar to blend the frames with
*/
void OpticalFlowCalcSDR::warpFramesForBlending(float fScalar) {
	// Calculate the blend scalar
	const float frameScalar12 = fScalar;
	const float frameScalar21 = static_cast<float>(1.0) - fScalar;

	// Calculate variables so the threds don't have to do it
	const int directionIdxOffset = m_iDimY * m_dResolutionDivider * m_iDimX * m_dResolutionDivider;
	const int channelIdxOffset = static_cast<unsigned int>(m_iDimY * m_iDimX);

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
		warpFrameKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
														          m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU, 
																  m_warpedFrame12.arrayPtrGPU, frameScalar12, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, channelIdxOffset);
		artifactRemovalKernelForBlending << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																				      m_warpedFrame12.arrayPtrGPU, m_iDimY, m_iDimX);

		// Frame 2 to Frame 1
		warpFrameKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU, 
																  m_warpedFrame21.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, channelIdxOffset);
		artifactRemovalKernelForBlending << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																					  m_warpedFrame21.arrayPtrGPU, m_iDimY, m_iDimX);

	} else {
		// Frame 1 to Frame 2
		warpFrameKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																  m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_warpedFrame12.arrayPtrGPU, frameScalar12, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, channelIdxOffset);
		artifactRemovalKernelForBlending << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame2.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																					  m_warpedFrame12.arrayPtrGPU, m_iDimY, m_iDimX);

		// Frame 2 to Frame 1
		warpFrameKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_warpedFrame21.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, channelIdxOffset);
		artifactRemovalKernelForBlending << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame1.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																					  m_warpedFrame21.arrayPtrGPU, m_iDimY, m_iDimX);

	}

	// Synchronize streams to ensure completion
	cudaStreamSynchronize(warpStream1);
	cudaStreamSynchronize(warpStream2);

	// Clean up streams
	cudaStreamDestroy(warpStream1);
	cudaStreamDestroy(warpStream2);
}

/*
* Blends warpedFrame1 to warpedFrame2
*
* @param dScalar: The scalar to blend the frames with
*/
void OpticalFlowCalcSDR::blendFrames(float fScalar) {
	// Calculate the blend scalar
	const float frame1Scalar = static_cast<float>(1.0) - fScalar;
	const float frame2Scalar = fScalar;
	const int scaledDimX = static_cast<unsigned int>(m_iDimX * m_dDimScalar);

	// Blend the frames
	blendFrameKernelSDR << <m_grid, m_threads2 >> >(m_warpedFrame12.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
												 m_outputFrame.arrayPtrGPU, frame1Scalar, frame2Scalar,
	                                             m_iDimY, m_iDimX, scaledDimX);
}

/*
* Draws the flow as an RGB image
*
* @param blendScalar: The scalar that determines how much of the source frame is blended with the flow
*/
void OpticalFlowCalcSDR::drawFlowAsHSV(const double blendScalar) const {
	// Calculate variables so the threds don't have to do it
	const int directionIdxOffset = m_iDimY * m_dResolutionDivider * m_iDimX * m_dResolutionDivider;
	const int scaledDimX = static_cast<unsigned int>(m_iDimX * m_dDimScalar);

	if (m_bBisNewest) {
		convertFlowToHSVKernelSDR << <m_grid, m_threads2 >> > (m_blurredOffsetArray12.arrayPtrGPU, m_outputFrame.arrayPtrGPU,
														    m_frame2.arrayPtrGPU, blendScalar, m_iLowDimX, m_iDimY, m_iDimX, 
															m_dResolutionDivider, directionIdxOffset, scaledDimX);
	} else {
		convertFlowToHSVKernelSDR << <m_grid, m_threads2 >> > (m_blurredOffsetArray12.arrayPtrGPU, m_outputFrame.arrayPtrGPU,
														    m_frame1.arrayPtrGPU, blendScalar, m_iLowDimX, m_iDimY, m_iDimX, 
															m_dResolutionDivider, directionIdxOffset, scaledDimX);
	}
}