#include <amvideo.h>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include "opticalFlowCalcSDR.cuh"

// Kernel that converts an NV12 array to a P010 array
__global__ void convertNV12toP010KernelSDR(const unsigned char* nv12Array, unsigned short* p010Array, const unsigned int dimY,
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

// Kernel that calculates the absolute difference between two frames using the offset array
__global__ void calcImageDeltaSDR(const unsigned char* frame1, const unsigned char* frame2, unsigned char* imageDeltaArray,
							   const int* offsetArray, const int dimZ, const int dimY, const int dimX, const double dResolutionScalar) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	if (cz < dimZ && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int absIndex = cz * dimY * dimX + cy * dimX + cx;
		const int offsetX = -offsetArray[absIndex];
		const int offsetY = -offsetArray[dimZ * dimY * dimX + absIndex];

		// Current pixel is outside of frame
		if ((cy * dResolutionScalar + offsetY < 0) || (cx * dResolutionScalar + offsetX < 0) || 
			(cy * dResolutionScalar + offsetY >= dimY * dResolutionScalar) || 
			(cx * dResolutionScalar + offsetX >= dimX * dResolutionScalar)) {
			imageDeltaArray[absIndex] = 0;
		// Current pixel is inside of frame
		} else {
			const int newCx = static_cast<int>(fmin(fmax(cx * dResolutionScalar + offsetX, 0.0), static_cast<double>(dimX) * dResolutionScalar - 1.0));
			const int newCy = static_cast<int>(fmin(fmax(cy * dResolutionScalar + offsetY, 0.0), static_cast<double>(dimY) * dResolutionScalar - 1.0));
			imageDeltaArray[absIndex] = abs(frame1[newCy * static_cast<unsigned int>(dimX * dResolutionScalar) + newCx] - 
				frame2[static_cast<unsigned int>(cy * dResolutionScalar) * static_cast<unsigned int>(dimX * dResolutionScalar) + static_cast<unsigned int>(cx * dResolutionScalar)]);
		}
	}
}

// Kernel that sums up all the pixel deltas of each window
__global__ void calcDeltaSumsSDR(unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int windowDimY,
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

// Kernel that warps a frame according to the offset array
__global__ void warpFrameKernelForOutputSDR(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones,
									     unsigned short* warpedFrame, const float frameScalar, const int dimY, const int dimX, 
									     const double dResolutionDivider, const double dDimScalar) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(dimY * dResolutionDivider * dimX * dResolutionDivider) + static_cast<unsigned int>(cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)]) * frameScalar);

		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			warpedFrame[newCy * static_cast<unsigned int>(dimX * dDimScalar) + newCx] = static_cast<unsigned short>(frame1[cy * dimX + cx]) << 8;
			atomicAdd(&hitCount[newCy * dimX + newCx], ones[cy * dimX + cx]);
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(2 * cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>((cx / 2) * 2 * dResolutionDivider)]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(dimY * dResolutionDivider * dimX * dResolutionDivider) + static_cast<unsigned int>(2 * cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>((cx / 2) * 2 * dResolutionDivider)]) * frameScalar / 2.0);

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
__global__ void warpFrameKernelForBlendingSDR(const unsigned char* frame1, const int* offsetArray, int* hitCount, int* ones,
										   unsigned char* warpedFrame, const float frameScalar, const int dimY, const int dimX, 
										   const double dResolutionDivider) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)]) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(dimY * dResolutionDivider * dimX * dResolutionDivider) + static_cast<unsigned int>(cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)]) * frameScalar);

		// Check if the current pixel is inside the frame
		if ((cy + offsetY >= 0) && (cy + offsetY < dimY) && (cx + offsetX >= 0) && (cx + offsetX < dimX)) {
			const int newCx = fminf(fmaxf(cx + offsetX, 0), dimX - 1);
			const int newCy = fminf(fmaxf(cy + offsetY, 0), dimY - 1);
			warpedFrame[newCy * dimX + newCx] = frame1[cy * dimX + cx];
			atomicAdd(&hitCount[newCy * dimX + newCx], ones[cy * dimX + cx]);
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[static_cast<unsigned int>(2 * cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>((cx / 2) * 2 * dResolutionDivider)]) * frameScalar);
		const int offsetY = static_cast<int>((static_cast<float>(offsetArray[static_cast<unsigned int>(dimY * dResolutionDivider * dimX * dResolutionDivider) + static_cast<unsigned int>(2 * cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>((cx / 2) * 2 * dResolutionDivider)]) * frameScalar / 2.0));

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
__global__ void artifactRemovalKernelForOutputSDR(const unsigned char* frame1, const int* hitCount, unsigned short* warpedFrame,
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
__global__ void artifactRemovalKernelForBlendingSDR(const unsigned char* frame1, const int* hitCount, unsigned char* warpedFrame,
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
__global__ void blendFrameKernelSDR(const unsigned char* frame1, const unsigned char* frame2, unsigned short* blendedFrame,
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
__global__ void convertFlowToHSVKernelSDR(const int* flowArray, unsigned short* p010Array, const unsigned char* frame1,
                                       const unsigned int dimY, const unsigned int dimX, const float saturation,
                                       const float value, const double dResolutionDivider, const double dDimScalar) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Blend scalar
	constexpr float scalar = 0.5;

	// Get the current flow values
	const double x = flowArray[static_cast<unsigned int>(cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)];
	const double y = flowArray[static_cast<unsigned int>(dimY * dResolutionDivider * dimX * dResolutionDivider) + static_cast<unsigned int>(cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)];

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

	// Prevent random colors when there is no flow
	if (fabs(x) < 1.0 && fabs(y) < 1.0) {
		rgb = { 0, 0, 0 };
	}

	// Write the converted RGB values to the array
	if (cz < 2 && cy < dimY && cx < dimX) {
		// Y Channel
		if (cz == 0) {
			p010Array[cy * static_cast<unsigned int>(dimX * dDimScalar) + cx] = static_cast<unsigned short>((fmaxf(fminf(static_cast<float>(0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b), 255.0), 0.0)) * scalar + frame1[cy * dimX + cx] * (1.0 - scalar)) << 8;
		// U Channel
		} else if (cz == 1 && cx % 2 == 0 && cy < dimY / 2) {
			p010Array[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + (cx / 2) * 2] = static_cast<unsigned short>(fmaxf(fminf(static_cast<float>(0.492 * (rgb.b - (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b)) + 128), 255.0), 0.0)) << 8;
		// V Channel
		} else if (cy < dimY / 2) {
			p010Array[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + (cx / 2) * 2 + 1] = static_cast<unsigned short>(fmaxf(fminf(static_cast<float>(0.877 * (rgb.r - (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b)) + 128), 255.0), 0.0)) << 8;
		}
	}
}

/*
* Initializes the SDR optical flow calculator
*
* @param dimY: The height of the frame
* @param dimX: The width of the frame
* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
* @param dResolutionScalar: The scalar to scale the resolution with
*/
OpticalFlowCalcSDR::OpticalFlowCalcSDR(const unsigned int dimY, const unsigned int dimX, const double dDimScalar, const double dResolutionScalar) {
	m_dResolutionScalar = dResolutionScalar;
	m_dResolutionDivider = 1.0 / dResolutionScalar;
	m_iDimX = dimX;
	m_iDimY = dimY;
	m_iLowDimX = static_cast<unsigned int>(static_cast<double>(dimX) * m_dResolutionDivider);
	m_iLowDimY = static_cast<unsigned int>(static_cast<double>(dimY) * m_dResolutionDivider);
	m_iLowDimZ = 5;
	m_dDimScalar = dDimScalar;
	m_lowGrid.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / static_cast<double>(NUM_THREADS)), 1.0));
	m_lowGrid.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / static_cast<double>(NUM_THREADS)), 1.0));
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
	m_grid.x = static_cast<int>(fmax(ceil(dimX / static_cast<double>(NUM_THREADS)), 1.0));
	m_grid.y = static_cast<int>(fmax(ceil(dimY / static_cast<double>(NUM_THREADS)), 1.0));
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
	convertNV12toP010KernelSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}

	// Download the output frame
	m_outputFrame.download(pOutBuffer);
}

/*
* Calculates the optical flow between frame1 and frame2
*
* @param iNumIterations: Number of iterations to calculate the optical flow
* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
*/
void OpticalFlowCalcSDR::calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps) {
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
				calcImageDeltaSDR << <m_lowGrid, m_threads5 >> > (m_frame1.arrayPtrGPU, m_frame2.arrayPtrGPU,
															   m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
															   m_iLowDimZ, m_iLowDimY, m_iLowDimX, m_dResolutionScalar);
			} else {
				calcImageDeltaSDR << <m_lowGrid, m_threads5 >> > (m_frame2.arrayPtrGPU, m_frame1.arrayPtrGPU,
															   m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
															   m_iLowDimZ, m_iLowDimY, m_iLowDimX, m_dResolutionScalar);
			}

			// 2. Sum up the deltas of each window
			calcDeltaSumsSDR << <m_lowGrid, m_threads5 >> > (m_imageDeltaArray.arrayPtrGPU, m_summedUpDeltaArray.arrayPtrGPU,
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
* Warps the frames according to the calculated optical flow
*
* @param fScalar: The scalar to blend the frames with
* @param bOutput12: Whether to output the warped frame 12 or 21
*/
void OpticalFlowCalcSDR::warpFramesForOutput(float fScalar, const bool bOutput12) {
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
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
														          m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU, 
																  m_outputFrame.arrayPtrGPU, frameScalar12, m_iDimY, 
																  m_iDimX, m_dResolutionDivider, m_dDimScalar);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		// Frame 2 to Frame 1
		} else {
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU, 
																  m_outputFrame.arrayPtrGPU, frameScalar21, m_iDimY, 
																  m_iDimX, m_dResolutionDivider, m_dDimScalar);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		}
	} else {
		// Frame 1 to Frame 2
		if (bOutput12) {
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																  m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_outputFrame.arrayPtrGPU, frameScalar12, m_iDimY,
																  m_iDimX, m_dResolutionDivider, m_dDimScalar);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																		m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		// Frame 2 to Frame 1
		} else {
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_outputFrame.arrayPtrGPU, frameScalar21, m_iDimY,
																  m_iDimX, m_dResolutionDivider, m_dDimScalar);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
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
void OpticalFlowCalcSDR::warpFramesForBlending(float fScalar) {
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
		warpFrameKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																			    m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU, 
																			    m_warpedFrame12.arrayPtrGPU, frameScalar12, m_iDimY, 
																			    m_iDimX, m_dResolutionDivider);
		artifactRemovalKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																				      m_warpedFrame12.arrayPtrGPU, m_iDimY, m_iDimX);

		// Frame 2 to Frame 1
		warpFrameKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																			    m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU, 
																				m_warpedFrame21.arrayPtrGPU, frameScalar21, m_iDimY, 
																				m_iDimX, m_dResolutionDivider);
		artifactRemovalKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																					  m_warpedFrame21.arrayPtrGPU, m_iDimY, m_iDimX);

	} else {
		// Frame 1 to Frame 2
		warpFrameKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																				m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU,
																				m_warpedFrame12.arrayPtrGPU, frameScalar12, m_iDimY,
																				m_iDimX, m_dResolutionDivider);
		artifactRemovalKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream1 >> > (m_frame2.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																					  m_warpedFrame12.arrayPtrGPU, m_iDimY, m_iDimX);

		// Frame 2 to Frame 1
		warpFrameKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																				m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU,
																				m_warpedFrame21.arrayPtrGPU, frameScalar21, m_iDimY,
																				m_iDimX, m_dResolutionDivider);
		artifactRemovalKernelForBlendingSDR << <m_grid, m_threads2, 0, warpStream2 >> > (m_frame1.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
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
void OpticalFlowCalcSDR::blendFrames(float fScalar) {
	// Calculate the blend scalar
	const float frame1Scalar = static_cast<float>(1.0) - fScalar;
	const float frame2Scalar = fScalar;

	// Blend the frames
	blendFrameKernelSDR << <m_grid, m_threads2 >> >(m_warpedFrame12.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
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
void OpticalFlowCalcSDR::drawFlowAsHSV(const float saturation, const float value) const {
	if (m_bBisNewest) {
		convertFlowToHSVKernelSDR << <m_grid, m_threads2 >> > (m_blurredOffsetArray12.arrayPtrGPU, m_outputFrame.arrayPtrGPU,
														    m_frame2.arrayPtrGPU, m_iDimY, m_iDimX, 
															saturation, value, m_dResolutionDivider, m_dDimScalar);
	} else {
		convertFlowToHSVKernelSDR << <m_grid, m_threads2 >> > (m_blurredOffsetArray12.arrayPtrGPU, m_outputFrame.arrayPtrGPU,
														    m_frame1.arrayPtrGPU, m_iDimY, m_iDimX, 
															saturation, value, m_dResolutionDivider, m_dDimScalar);
	}

	// Check for CUDA errors
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaError));
		exit(-1);
	}
}