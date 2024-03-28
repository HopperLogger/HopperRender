#include <amvideo.h>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include "opticalFlowCalcHDR.cuh"

// Kernel that scales a P010 frame to a P010 frame for madVR
__global__ void scaleFrameKernelHDR(const unsigned short* sourceFrame, unsigned short* outputFrame, const unsigned int dimY,
	const unsigned int dimX, const unsigned int scaledDimY, const unsigned int scaledDimX, const unsigned int channelIdxOffset,
	const unsigned int scaledChannelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Check if the current thread is inside the Y-Channel or the U/V-Channel
	if (cx < scaledDimX && ((cz == 0 && cy < scaledDimY) || (cz == 1 && cy < (scaledDimY >> 1)))) {
		outputFrame[cz * scaledChannelIdxOffset + cy * scaledDimX + cx] = sourceFrame[cz * channelIdxOffset + cy * dimX + cx];
	}
}

// Kernel that blurs a frame
__global__ void blurFrameKernelHDR(const unsigned short* frameArray, unsigned short* blurredFrameArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char lumStart,
								const unsigned char lumEnd, const unsigned short lumPixelCount, const char chromStart, 
								const unsigned char chromEnd, const unsigned short chromPixelCount, const unsigned short dimY, const unsigned short dimX) {
	// Shared memory for the frame to prevent multiple global memory accesses
	extern __shared__ unsigned short sharedFrameArray[];

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
__global__ void warpFrameKernelForOutputHDR(const unsigned short* frame1, const int* offsetArray, int* hitCount, int* ones,
											unsigned short* warpedFrame, const float frameScalar, const unsigned int lowDimY, const unsigned int lowDimX,
											const unsigned int dimY, const unsigned int dimX, const float resolutionDivider,
											const unsigned int directionIdxOffset, const unsigned int scaledDimX, 
										    const unsigned int channelIdxOffset, const unsigned int scaledChannelIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	const int scaledCx = static_cast<int>(static_cast<float>(cx) * resolutionDivider); // The X-Index of the current thread in the offset array
	const int scaledCy = static_cast<int>(static_cast<float>(cy) * resolutionDivider); // The Y-Index of the current thread in the offset array

	// Y-Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[scaledCy * lowDimX + scaledCx]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + scaledCy * lowDimX + scaledCx]) * frameScalar);
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < dimY && newCx >= 0 && newCx < dimX) {
			warpedFrame[newCy * scaledDimX + newCx] = frame1[cy * dimX + cx];
			atomicAdd(&hitCount[newCy * dimX + newCx], ones[cy * dimX + cx]);
		}

	// U/V-Channel
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[(scaledCy << 1) * lowDimX + (scaledCx & ~1)]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + (scaledCy << 1) * lowDimX + (scaledCx & ~1)]) * frameScalar) >> 1;
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < (dimY >> 1) && newCx >= 0 && newCx < dimX) {
			// U-Channel
			if ((cx & 1) == 0) {
				warpedFrame[scaledChannelIdxOffset + newCy * scaledDimX + (newCx & ~1)] = frame1[channelIdxOffset + cy * dimX + cx];

			// V-Channel
			} else {
				warpedFrame[scaledChannelIdxOffset + newCy * scaledDimX + (newCx & ~1) + 1] = frame1[channelIdxOffset + cy * dimX + cx];
			}
		}
	}
}

// Kernel that warps a frame according to the offset array
__global__ void warpFrameKernelForBlendingHDR(const unsigned short* frame1, const int* offsetArray, int* hitCount, int* ones,
										      unsigned short* warpedFrame, const float frameScalar, const unsigned int lowDimY, const unsigned int lowDimX,
											  const unsigned int dimY, const unsigned int dimX, const float resolutionDivider,
											  const unsigned int directionIdxOffset, const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = blockIdx.z * blockDim.z + threadIdx.z;

	const int scaledCx = static_cast<int>(static_cast<float>(cx) * resolutionDivider); // The X-Index of the current thread in the offset array
	const int scaledCy = static_cast<int>(static_cast<float>(cy) * resolutionDivider); // The Y-Index of the current thread in the offset array

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
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[(scaledCy << 1) * lowDimX + (scaledCx & ~1)]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + (scaledCy << 1) * lowDimX + (scaledCx & ~1)]) * frameScalar) >> 1;
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < (dimY >> 1) && newCx >= 0 && newCx < dimX) {
			// U-Channel
			if ((cx & 1) == 0) {
				warpedFrame[channelIdxOffset + newCy * dimX + (newCx & ~1)] = frame1[channelIdxOffset + cy * dimX + cx];

			// V-Channel
			} else {
				warpedFrame[channelIdxOffset + newCy * dimX + (newCx & ~1) + 1] = frame1[channelIdxOffset + cy * dimX + cx];
			}
		}
	}
}

// Kernel that removes artifacts from the warped frame
__global__ void artifactRemovalKernelForOutputHDR(const unsigned short* frame1, const int* hitCount, unsigned short* warpedFrame,
	const unsigned int dimY, const unsigned int dimX, const int scaledDimX, 
	const unsigned int channelIdxOffset, const unsigned int scaledChannelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int threadIndex2D = cy * dimX + cx; // Standard thread index without Z-Dim

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[cy * scaledDimX + cx] = frame1[threadIndex2D];
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[scaledChannelIdxOffset + cy * scaledDimX + cx] = frame1[channelIdxOffset + threadIndex2D];
		}
	}
}

// Kernel that blends frame1 to frame2
__global__ void blendFrameKernelHDR(const unsigned short* frame1, const unsigned short* frame2, unsigned short* blendedFrame,
	const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
	const unsigned int dimX, const int scaledDimX, const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		blendedFrame[cy * scaledDimX + cx] = 
			static_cast<float>(frame1[cy * dimX + cx]) * frame1Scalar + 
			static_cast<float>(frame2[cy * dimX + cx]) * frame2Scalar;
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		blendedFrame[dimY * scaledDimX + cy * scaledDimX + cx] = 
			static_cast<float>(frame1[channelIdxOffset + cy * dimX + cx]) * frame1Scalar + 
			static_cast<float>(frame2[channelIdxOffset + cy * dimX + cx]) * frame2Scalar;
	}
}

// Kernel that creates an HSV flow image from the offset array
__global__ void convertFlowToHSVKernelHDR(const int* flowArray, unsigned short* p010Array, const unsigned short* frame1,
                                          const double blendScalar, const int lowDimX, const unsigned int dimY, const unsigned int dimX, 
										  const double resolutionDivider, const int directionIdxOffset, const int scaledDimX,
										  const unsigned int scaledChannelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	const unsigned int scaledCx = static_cast<unsigned int>(static_cast<float>(cx) * resolutionDivider); // The X-Index of the current thread in the offset array
	const unsigned int scaledCy = static_cast<unsigned int>(static_cast<float>(cy) * resolutionDivider); // The Y-Index of the current thread in the offset array

	// Get the current flow values
	float x;
	float y;
	if (cz == 0) {
		x = flowArray[scaledCy * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + scaledCy * lowDimX + scaledCx];
	} else {
		x = flowArray[(scaledCy << 1) * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + (scaledCy << 1) * lowDimX + scaledCx];
	}

	// RGB struct
	struct RGB {
		int r, g, b;
	};

	// Calculate the angle in radians
	const float angle_rad = std::atan2(y, x);

	// Convert radians to degrees
	float angle_deg = angle_rad * (180.0f / 3.14159265359f);

	// Ensure the angle is positive
	if (angle_deg < 0) {
		angle_deg += 360.0f;
	}

	// Normalize the angle to the range [0, 360]
	angle_deg = fmodf(angle_deg, 360.0f);
	if (angle_deg < 0) {
		angle_deg += 360.0f;
	}

	// Map the angle to the hue value in the HSV model
	const float hue = angle_deg / 360.0f;

	// Convert HSV to RGB
	const int h_i = static_cast<int>(hue * 6.0f);
	const float f = hue * 6.0f - h_i;
	const float q = 1.0f - f;

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
	if (fabsf(x) < 1.0f && fabsf(y) < 1.0f) {
		rgb = { 0, 0, 0 };
	}

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		p010Array[cy * scaledDimX + cx] = 
			(static_cast<unsigned short>(
				(fmaxf(fminf(0.299f * rgb.r + 0.587f * rgb.g + 0.114f * rgb.b, 255.0f), 0.0f)) * blendScalar) << 8) + 
				frame1[cy * dimX + cx] * (1.0f - blendScalar);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		// U Channel
		if ((cx & 1) == 0) {
			p010Array[scaledChannelIdxOffset + cy * scaledDimX + (cx & ~1)] = 
				static_cast<unsigned short>(
					fmaxf(fminf(0.492f * (rgb.b - (0.299f * rgb.r + 0.587f * rgb.g + 0.114f * rgb.b)) + 128.0f, 255.0f), 0.0f)
				) << 8;
		// V Channel
		} else {
			p010Array[scaledChannelIdxOffset + cy * scaledDimX + (cx & ~1) + 1] = 
				static_cast<unsigned short>(
					fmaxf(fminf(0.877f * (rgb.r - (0.299f * rgb.r + 0.587f * rgb.g + 0.114f * rgb.b)) + 128.0f, 255.0f), 0.0f)
				) << 8;
		}
	}
}

/*
* Initializes the HDR optical flow calculator
*
* @param dimY: The height of the frame
* @param dimX: The width of the frame
* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
* @param fResolutionDivider: The scalar to scale the resolution with
*/
OpticalFlowCalcHDR::OpticalFlowCalcHDR(const unsigned int dimY, const unsigned int dimX, const double dDimScalar, const float fResolutionDivider) {
	// Variables
	m_fResolutionDivider = fResolutionDivider;
	m_fResolutionScalar = 1.0f / fResolutionDivider;
	m_iDimX = dimX;
	m_iDimY = dimY;
	m_iLowDimX = static_cast<unsigned int>(static_cast<float>(dimX) * m_fResolutionDivider);
	m_iLowDimY = static_cast<unsigned int>(static_cast<float>(dimY) * m_fResolutionDivider);
	m_iNumLayers = 5;
	m_dDimScalar = dDimScalar;
	m_iDirectionIdxOffset = m_iNumLayers * m_iLowDimY * m_iLowDimX;
	m_iLayerIdxOffset = m_iLowDimY * m_iLowDimX;
	m_iChannelIdxOffset = m_iDimY * m_iDimX;
	m_iScaledChannelIdxOffset = static_cast<unsigned int>(m_iDimY * m_iDimX * m_dDimScalar);
	m_iScaledDimX = static_cast<unsigned int>(m_iDimX * m_dDimScalar);
	m_iScaledDimY = static_cast<unsigned int>(m_iDimY * m_dDimScalar);

	// Girds
	m_lowGrid32x32x1.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / 32.0), 1.0));
	m_lowGrid32x32x1.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / 32.0), 1.0));
	m_lowGrid32x32x1.z = 1;
	m_lowGrid16x16x4.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / 16.0), 1.0));
	m_lowGrid16x16x4.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / 16.0), 1.0));
	m_lowGrid16x16x4.z = 4;
	m_lowGrid16x16x1.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / 16.0), 1.0));
	m_lowGrid16x16x1.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / 16.0), 1.0));
	m_lowGrid16x16x1.z = 1;
	m_lowGrid8x8x1.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / 8.0), 1.0));
	m_lowGrid8x8x1.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / 8.0), 1.0));
	m_lowGrid8x8x1.z = 1;
	m_grid16x16x1.x = static_cast<int>(fmax(ceil(dimX / 16.0), 1.0));
	m_grid16x16x1.y = static_cast<int>(fmax(ceil(dimY / 16.0), 1.0));
	m_grid16x16x1.z = 1;
	m_grid8x8x1.x = static_cast<int>(fmax(ceil(dimX / 8.0), 1.0));
	m_grid8x8x1.y = static_cast<int>(fmax(ceil(dimY / 8.0), 1.0));
	m_grid8x8x1.z = 1;

	// Threads
	m_threads32x32x1.x = 32;
	m_threads32x32x1.y = 32;
	m_threads32x32x1.z = 1;
	m_threads16x16x2.x = 16;
	m_threads16x16x2.y = 16;
	m_threads16x16x2.z = 2;
	m_threads16x16x1.x = 16;
	m_threads16x16x1.y = 16;
	m_threads16x16x1.z = 1;
	m_threads8x8x5.x = 8;
	m_threads8x8x5.y = 8;
	m_threads8x8x5.z = 5;
	m_threads8x8x2.x = 8;
	m_threads8x8x2.y = 8;
	m_threads8x8x2.z = 2;

	// GPU Arrays
	m_frame1.init({ 1, dimY, dimX }, 0, static_cast<size_t>(3.0 * static_cast<double>(dimY * dimX)));
	m_frame2.init({ 1, dimY, dimX }, 0, static_cast<size_t>(3.0 * static_cast<double>(dimY * dimX)));
	m_blurredFrame1.init({ 1, dimY, dimX }, 0, static_cast<size_t>(3.0 * static_cast<double>(dimY * dimX)));
	m_blurredFrame2.init({ 1, dimY, dimX }, 0, static_cast<size_t>(3.0 * static_cast<double>(dimY * dimX)));
	m_offsetArray12.init({ 2, 5, dimY, dimX });
	m_offsetArray21.init({ 2, dimY, dimX });
	m_blurredOffsetArray12.init({ 2, dimY, dimX });
	m_blurredOffsetArray21.init({ 2, dimY, dimX });
	m_statusArray.init({ dimY, dimX });
	m_summedUpDeltaArray.init({ 5, dimY, dimX });
	m_normalizedDeltaArray.init({ 5, dimY, dimX });
	m_lowestLayerArray.init({ dimY, dimX });
	m_warpedFrame12.init({ 1, dimY, dimX }, 0, static_cast<size_t>(3.0 * static_cast<double>(dimY * dimX)));
	m_warpedFrame21.init({ 1, dimY, dimX }, 0, static_cast<size_t>(3.0 * static_cast<double>(dimY * dimX)));
	m_outputFrame.init({ 1, dimY, dimX }, 0, static_cast<size_t>(3.0 * dimY * dimX * dDimScalar));
	m_hitCount12.init({ 1, dimY, dimX });
	m_hitCount21.init({ 1, dimY, dimX });
	m_ones.init({ 1, dimY, dimX }, 1);
}

/*
* Updates the frame1 array
*
* @param pInBuffer: Pointer to the input frame
*/
void OpticalFlowCalcHDR::updateFrame1(const unsigned char* pInBuffer) {
	m_frame1.fillData(pInBuffer);
	m_bBisNewest = false;
}

/*
* Updates the frame2 array
*
* @param pInBuffer: Pointer to the input frame
*/
void OpticalFlowCalcHDR::updateFrame2(const unsigned char* pInBuffer) {
	m_frame2.fillData(pInBuffer);
	m_bBisNewest = true;
}

/*
* Copies the frame in the correct format to the output buffer
*
* @param pInBuffer: Pointer to the input frame
* @param pOutBuffer: Pointer to the output frame
*/
void OpticalFlowCalcHDR::copyFrame(const unsigned char* pInBuffer, unsigned char* pOutBuffer) {
	if (m_dDimScalar == 1.0) {
		memcpy(pOutBuffer, pInBuffer, 3ll * m_iDimY * m_iDimX);
	} else {
		// Set the array entries to the provided value
		m_frame1.fillData(pInBuffer);

		// Convert the NV12 frame to P010
		scaleFrameKernelHDR << <m_grid8x8x1, m_threads8x8x2 >> > (m_frame1.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iScaledDimY, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);

		// Download the output frame
		m_outputFrame.download(pOutBuffer);
	}
}

/*
* Blurs a frame
*
* @param kernelSize: Size of the kernel to use for the blur
* @param directOutput: Whether to output the blurred frame directly
*/
void OpticalFlowCalcHDR::blurFrameArray(const unsigned char kernelSize, const bool directOutput) {
	const unsigned char boundsOffset = kernelSize >> 1;
	const unsigned char chacheSize = kernelSize + (boundsOffset << 1);
	const size_t sharedMemSize = chacheSize * chacheSize * sizeof(unsigned short);
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
			blurFrameKernelHDR << <gridBF, threadsBF, sharedMemSize >> > (m_frame1.arrayPtrGPU, m_blurredFrame1.arrayPtrGPU, kernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, lumStart, lumEnd, lumPixelCount, chromStart, chromEnd, chromPixelCount, m_iDimY, m_iDimX);
		}

		// Convert the NV12 frame to P010 if we are doing direct output
		if (directOutput) {
			scaleFrameKernelHDR << <m_grid8x8x1, m_threads8x8x2 >> > (m_blurredFrame1.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iScaledDimY, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);
		}
	} else {
		// No need to blur the frame if the kernel size is less than 4
		if (kernelSize < 4) {
			cudaMemcpy(m_blurredFrame2.arrayPtrGPU, m_frame2.arrayPtrGPU, m_frame2.bytes, cudaMemcpyDeviceToDevice);
		} else {
			// Launch kernel
			blurFrameKernelHDR << <gridBF, threadsBF, sharedMemSize >> > (m_frame2.arrayPtrGPU, m_blurredFrame2.arrayPtrGPU, kernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, lumStart, lumEnd, lumPixelCount, chromStart, chromEnd, chromPixelCount, m_iDimY, m_iDimX);
		}

		// Convert the NV12 frame to P010 if we are doing direct output
		if (directOutput) {
			scaleFrameKernelHDR << <m_grid8x8x1, m_threads8x8x2 >> > (m_blurredFrame2.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iScaledDimY, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);
		}
	}
}

/*
* Calculates the optical flow between frame1 and frame2
*
* @param iNumIterations: Number of iterations to calculate the optical flow
* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
*/
void OpticalFlowCalcHDR::calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps) {
	// Reset variables
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
	unsigned int NUM_BLOCKS_X = max(static_cast<int>(ceil(static_cast<float>(m_iLowDimX) / num_threads)), 1);
	unsigned int NUM_BLOCKS_Y = max(static_cast<int>(ceil(static_cast<float>(m_iLowDimY) / num_threads)), 1);

	// Use dim3 structs for block and grid size
	dim3 gridCDS(NUM_BLOCKS_X, NUM_BLOCKS_Y, 5);
	dim3 threadsCDS(num_threads, num_threads, 1);

	// Set layer 0 of the X-Dir to 0
	cudaMemset(m_offsetArray12.arrayPtrGPU, 0, m_iLayerIdxOffset * sizeof(int));
	// Set layers 0-5 of the Y-Dir to 0
	cudaMemset(m_offsetArray12.arrayPtrGPU + m_iDirectionIdxOffset, 0, m_iDirectionIdxOffset * sizeof(int));
	// Set layers 1-4 of the X-Dir to -2,-1,1,2
	setInitialOffset << <m_lowGrid16x16x4, m_threads16x16x1 >> > (m_offsetArray12.arrayPtrGPU, m_iNumLayers, m_iLowDimY, m_iLowDimX, m_iLayerIdxOffset);

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		switch (iter) {
			case 0: iNumStepsPerIter = iNumSteps; break;
			case 1: iNumStepsPerIter = iNumSteps; break;
			case 2: iNumStepsPerIter = iNumSteps; break;
			case 3: iNumStepsPerIter = max(iNumSteps / 2, 1); break;
			case 4: iNumStepsPerIter = max(iNumSteps / 2, 1); break;
			case 5: iNumStepsPerIter = max(iNumSteps / 2, 1); break;
			case 6: iNumStepsPerIter = max(iNumSteps / 3, 1); break;
			case 7: iNumStepsPerIter = max(iNumSteps / 3, 1); break;
			case 8: iNumStepsPerIter = max(iNumSteps / 6, 1); break;
			default: iNumStepsPerIter = 1; break;
		}

		// Each step we adjust the offset array to find the ideal offset
		for (unsigned int step = 0; step < iNumStepsPerIter; step++) {
			// Reset the summed up delta array
			m_summedUpDeltaArray.zero();

			// 1. Calculate the image delta and sum up the deltas of each window
			if (windowDim >= 8) {
				calcDeltaSums8x8 << <gridCDS, threadsCDS, sharedMemSize>> > (m_summedUpDeltaArray.arrayPtrGPU, 
																	m_bBisNewest ? m_blurredFrame1.arrayPtrGPU : m_blurredFrame2.arrayPtrGPU,
                                                                 m_bBisNewest ? m_blurredFrame2.arrayPtrGPU : m_blurredFrame1.arrayPtrGPU,
																m_offsetArray12.arrayPtrGPU, m_iLayerIdxOffset, m_iDirectionIdxOffset,
																	m_iDimY, m_iDimX, m_iLowDimY, m_iLowDimX, windowDim, m_fResolutionScalar);
			} else if (windowDim == 4) {
				calcDeltaSums4x4 << <gridCDS, threadsCDS, sharedMemSize>> > (m_summedUpDeltaArray.arrayPtrGPU, 
																	m_bBisNewest ? m_blurredFrame1.arrayPtrGPU : m_blurredFrame2.arrayPtrGPU,
                                                                 m_bBisNewest ? m_blurredFrame2.arrayPtrGPU : m_blurredFrame1.arrayPtrGPU,
																m_offsetArray12.arrayPtrGPU, m_iLayerIdxOffset, m_iDirectionIdxOffset,
																	m_iDimY, m_iDimX, m_iLowDimY, m_iLowDimX, windowDim, m_fResolutionScalar);
			} else if (windowDim == 2) {
				calcDeltaSums2x2 << <gridCDS, threadsCDS, sharedMemSize>> > (m_summedUpDeltaArray.arrayPtrGPU, 
																	m_bBisNewest ? m_blurredFrame1.arrayPtrGPU : m_blurredFrame2.arrayPtrGPU,
                                                                 m_bBisNewest ? m_blurredFrame2.arrayPtrGPU : m_blurredFrame1.arrayPtrGPU,
																m_offsetArray12.arrayPtrGPU, m_iLayerIdxOffset, m_iDirectionIdxOffset,
																	m_iDimY, m_iDimX, m_iLowDimY, m_iLowDimX, windowDim, m_fResolutionScalar);
			} else if (windowDim == 1) {
				calcDeltaSums1x1 << <m_lowGrid8x8x1, m_threads8x8x5, sharedMemSize>> > (m_summedUpDeltaArray.arrayPtrGPU, 
																	m_bBisNewest ? m_blurredFrame1.arrayPtrGPU : m_blurredFrame2.arrayPtrGPU,
                                                                 m_bBisNewest ? m_blurredFrame2.arrayPtrGPU : m_blurredFrame1.arrayPtrGPU,
																m_offsetArray12.arrayPtrGPU, m_iLayerIdxOffset, m_iDirectionIdxOffset,
																	m_iDimY, m_iDimX, m_iLowDimY, m_iLowDimX, m_fResolutionScalar);
			}

			// 2. Normalize the summed up delta array and find the best layer
			normalizeDeltaSums << <m_lowGrid8x8x1, m_threads8x8x5 >> > (m_summedUpDeltaArray.arrayPtrGPU, m_lowestLayerArray.arrayPtrGPU,
				m_offsetArray12.arrayPtrGPU, windowDim, windowDim * windowDim,
				m_iDirectionIdxOffset, m_iLayerIdxOffset, m_iNumLayers, m_iLowDimY, m_iLowDimX);

			// 3. Adjust the offset array based on the comparison results
			adjustOffsetArray << <m_lowGrid32x32x1, m_threads32x32x1 >> > (m_offsetArray12.arrayPtrGPU, m_lowestLayerArray.arrayPtrGPU,
				m_statusArray.arrayPtrGPU, windowDim, m_iDirectionIdxOffset, m_iLayerIdxOffset,
				m_iNumLayers, m_iLowDimY, m_iLowDimX, step == iNumStepsPerIter - 1);
		}

		// 4. Adjust window size
		windowDim = max(windowDim >> 1, 1);
		num_threads = min(windowDim, 16);
		sharedMemSize = num_threads * num_threads * sizeof(unsigned int);
		NUM_BLOCKS_X = max(static_cast<int>(ceil(static_cast<float>(m_iLowDimX) / num_threads)), 1);
		NUM_BLOCKS_Y = max(static_cast<int>(ceil(static_cast<float>(m_iLowDimY) / num_threads)), 1);
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
void OpticalFlowCalcHDR::warpFramesForOutput(float fScalar, const bool bOutput12) {
	// Calculate the blend scalar
	const float frameScalar12 = fScalar;
	const float frameScalar21 = 1.0f - fScalar;

	// Reset the hit count array
	if (bOutput12) {
		m_hitCount12.zero();
	}
	else {
		m_hitCount21.zero();
	}

	// Frame 1 to Frame 2
	if (bOutput12) {
		warpFrameKernelForOutputHDR << <m_grid16x16x1, m_threads16x16x2 >> > (m_bBisNewest ? m_frame1.arrayPtrGPU : m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
			m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU,
			m_outputFrame.arrayPtrGPU, frameScalar12, m_iLowDimY, m_iLowDimX,
			m_iDimY, m_iDimX, m_fResolutionDivider, m_iLayerIdxOffset, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);
		artifactRemovalKernelForOutputHDR << <m_grid8x8x1, m_threads8x8x2 >> > (m_bBisNewest ? m_frame1.arrayPtrGPU : m_frame2.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
			m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);
	// Frame 2 to Frame 1
	} else {
		warpFrameKernelForOutputHDR << <m_grid16x16x1, m_threads16x16x2 >> > (m_bBisNewest ? m_frame2.arrayPtrGPU : m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
			m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU,
			m_outputFrame.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
			m_iDimY, m_iDimX, m_fResolutionDivider, m_iLayerIdxOffset, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);
		artifactRemovalKernelForOutputHDR << <m_grid8x8x1, m_threads8x8x2 >> > (m_bBisNewest ? m_frame2.arrayPtrGPU : m_frame1.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
			m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);
	}
}

/*
* Warps the frames according to the calculated optical flow
*
* @param fScalar: The scalar to blend the frames with
*/
void OpticalFlowCalcHDR::warpFramesForBlending(float fScalar) {
	// Calculate the blend scalar
	const float frameScalar12 = fScalar;
	const float frameScalar21 = 1.0f - fScalar;

	// Reset the hit count array
	m_hitCount12.zero();
	m_hitCount21.zero();

	// Create CUDA streams
	cudaStream_t warpStream1, warpStream2;
	cudaStreamCreate(&warpStream1);
	cudaStreamCreate(&warpStream2);

	// Frame 1 to Frame 2
	warpFrameKernelForBlendingHDR << <m_grid16x16x1, m_threads16x16x2, 0, warpStream1 >> > (m_bBisNewest ? m_frame1.arrayPtrGPU : m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
														        m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU, 
																m_warpedFrame12.arrayPtrGPU, frameScalar12, m_iLowDimY, m_iLowDimX,
																m_iDimY, m_iDimX, m_fResolutionDivider, m_iLayerIdxOffset, m_iChannelIdxOffset);
	artifactRemovalKernelForBlending << <m_grid8x8x1, m_threads8x8x2, 0, warpStream1 >> > (m_bBisNewest ? m_frame1.arrayPtrGPU : m_frame2.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																				    m_warpedFrame12.arrayPtrGPU, m_iDimY, m_iDimX, m_iChannelIdxOffset);

	// Frame 2 to Frame 1
	warpFrameKernelForBlendingHDR << <m_grid16x16x1, m_threads16x16x2, 0, warpStream2 >> > (m_bBisNewest ? m_frame2.arrayPtrGPU : m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU, 
																m_warpedFrame21.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
																m_iDimY, m_iDimX, m_fResolutionDivider, m_iLayerIdxOffset, m_iChannelIdxOffset);
	artifactRemovalKernelForBlending << <m_grid8x8x1, m_threads8x8x2, 0, warpStream2 >> > (m_bBisNewest ? m_frame2.arrayPtrGPU : m_frame1.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																					m_warpedFrame21.arrayPtrGPU, m_iDimY, m_iDimX, m_iChannelIdxOffset);


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
void OpticalFlowCalcHDR::blendFrames(float fScalar) {
	// Calculate the blend scalar
	const float frame1Scalar = 1.0f - fScalar;
	const float frame2Scalar = fScalar;

	// Blend the frames
	blendFrameKernelHDR << <m_grid16x16x1, m_threads16x16x2 >> > (m_warpedFrame12.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
		m_outputFrame.arrayPtrGPU, frame1Scalar, frame2Scalar,
		m_iDimY, m_iDimX, m_iScaledDimX, m_iChannelIdxOffset);
}

/*
* Draws the flow as an RGB image
*
* @param blendScalar: The scalar that determines how much of the source frame is blended with the flow
*/
void OpticalFlowCalcHDR::drawFlowAsHSV(const float blendScalar) const {
	convertFlowToHSVKernelHDR << <m_grid16x16x1, m_threads16x16x2 >> > (m_blurredOffsetArray12.arrayPtrGPU, m_outputFrame.arrayPtrGPU,
														m_bBisNewest ? m_frame2.arrayPtrGPU : m_frame1.arrayPtrGPU, blendScalar, m_iLowDimX, m_iDimY, m_iDimX, 
														m_fResolutionDivider, m_iLayerIdxOffset, m_iScaledDimX, m_iScaledChannelIdxOffset);
}