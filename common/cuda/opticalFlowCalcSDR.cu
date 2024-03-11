#include <amvideo.h>
#include <iomanip>
#include <vector>
#include <cuda_runtime_api.h>
#include "opticalFlowCalcSDR.cuh"

// Kernel that converts an NV12 array to a P010 array
__global__ void convertNV12toP010KernelSDR(const unsigned char* nv12Array, unsigned short* p010Array, const unsigned int dimY,
										   const unsigned int dimX, const double dimScalar) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	
	// Dimensions of the frame scaled for the renderer
	const unsigned int scaledDimX = static_cast<unsigned int>(dimX * dimScalar);
	const unsigned int scaledDimY = static_cast<unsigned int>(dimY * dimScalar);

	// Check if the current thread is inside the Y-Channel or the U/V-Channel
	if ((cz == 0 && cy < scaledDimY && cx < scaledDimX) || (cz == 1 && cy < (scaledDimY / 2) && cx < scaledDimX)) {
		p010Array[cz * dimY * scaledDimX + cy * scaledDimX + cx] = static_cast<unsigned short>(nv12Array[cz * dimY * dimX + cy * dimX + cx]) << 8;
	}
}

// Kernel that blurs a frame
__global__ void blurFrameKernel(const unsigned char* frameArray, unsigned char* blurredFrameArray, const int kernelSize, const int dimY,
						        const int dimX) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	if (kernelSize > 3) {
		// Calculate the x and y boundaries of the kernel
		int start = -(kernelSize / 2);
		int end = (kernelSize / 2);
		unsigned int blurredPixel = 0;

		// Collect the sum of the surrounding pixels
		// Y-Channel
		if (cz == 0 && cy < dimY && cx < dimX) {
			for (int y = start; y < end; y++) {
				for (int x = start; x < end; x++) {
					if ((cy + y) < dimY && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
						blurredPixel += frameArray[cz * dimY * dimX + (cy + y) * dimX + cx + x];
					} else {
						blurredPixel += frameArray[cz * dimY * dimX + cy * dimX + cx];
					}
				}
			}
			blurredPixel /= (end - start) * (end - start);
			blurredFrameArray[cz * dimY * dimX + cy * dimX + cx] = blurredPixel;
		// U/V-Channel
		} else if (cz == 1 && cy < dimY / 2 && cx < dimX) {
			start = -(kernelSize / 4);
			end = (kernelSize / 4);
			for (int y = start; y < end; y++) {
				for (int x = start; x < end; x++) {
					if ((cy + y) < dimY / 2 && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
						blurredPixel += frameArray[cz * dimY * dimX + (cy + y) * dimX + cx + x * 2];
					} else {
						blurredPixel += frameArray[cz * dimY * dimX + cy * dimX + cx];
					}
				}
			}
			blurredPixel /= (end - start) * (end - start);
			blurredFrameArray[cz * dimY * dimX + cy * dimX + cx] = blurredPixel;
		}
	} else {
		if ((cz == 0 && cy < dimY && cx < dimX) || (cz == 1 && cy < dimY / 2 && cx < dimX)) {
			blurredFrameArray[cz * dimY * dimX + cy * dimX + cx] = frameArray[cz * dimY * dimX + cy * dimX + cx];
		}
	}
}

// Kernel that calculates the absolute difference between two frames using the offset array
__global__ void calcImageDeltaSDR(const unsigned char* frame1, const unsigned char* frame2, unsigned char* imageDeltaArray,
							      const int* offsetArray, const int numLayers, const int lowDimY, const int lowDimX, 
								  const int dimY, const int dimX, const double resolutionScalar, const int directionIdxOffset, 
								  const int channelIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;
	const int channel = cz % 2; // YUV channel of the current thread

	// Is the current thread supposed to perform calculations
	if (cz < numLayers * 2 && cy < lowDimY && cx < lowDimX) {
		const int layer = cz / 2; // Layer of the current thread
		const int layerOffset = layer * lowDimY * lowDimX; // Offset to index the layer of the current thread
		const int scaledCx = static_cast<int>(cx * resolutionScalar); // The X-Index of the current thread in the input frames
		const int scaledCy = static_cast<int>(cy * resolutionScalar); // The Y-Index of the current thread in the input frames
		const int evenCx = (cx / 2) * 2; // The X-Index of the current thread rounded to be even

		const int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim
		const int threadIndex3D = cz * lowDimY * lowDimX + threadIndex2D; // Standard thread index

		// Y-Channel
		if (channel == 0) {
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
			const int newCx = scaledCx + (offsetX / 2) * 2;
			const int newCy = scaledCy * 0.5 + offsetY * 0.5;

			imageDeltaArray[threadIndex3D] = (newCy < 0 || newCx < 0 || newCy >= dimY / 2 || newCx >= dimX) ? 0 : 
				2 * abs(frame1[channelIdxOffset + newCy * dimX + newCx] - frame2[channelIdxOffset + static_cast<int>(scaledCy * 0.5) * dimX + scaledCx]);
		}
	}
}

// Kernel that sums up all the pixel deltas of each window
__global__ void calcDeltaSumsSDR(unsigned char* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int windowDimY,
							     const unsigned int windowDimX, const unsigned int numLayers, const unsigned int lowDimY, const unsigned int lowDimX) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int windowIndexX = cx / windowDimX;
	const unsigned int windowIndexY = cy / windowDimY;
	const int layer = cz / 2; // Layer of the current thread

	// Check if the thread is inside the frame
	if (cz < numLayers * 2 && cy < lowDimY && cx < lowDimX) {
		atomicAdd(&summedUpDeltaArray[layer * lowDimY * lowDimX + (windowIndexY * windowDimY) * lowDimX + (windowIndexX * windowDimX)],
			imageDeltaArray[cz * lowDimY * lowDimX + cy * lowDimX + cx]);
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
			if (cx % 2 == 0) {
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
				warpedFrame[dimY * dimX + newCy * dimX + (newCx / 2) * 2] = frame1[dimY * dimX + cy * dimX + cx];

			// V Channel
			} else {
				warpedFrame[dimY * dimX + newCy * dimX + (newCx / 2) * 2 + 1] = frame1[dimY * dimX + cy * dimX + cx];
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
			warpedFrame[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + cx] = static_cast<unsigned short>(frame1[dimY * dimX + cy * dimX + cx]) << 8;
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
			warpedFrame[dimY * dimX + cy * dimX + cx] = frame1[dimY * dimX + cy * dimX + cx];
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
		blendedFrame[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + cx] = static_cast<unsigned short>(static_cast<float>(frame1[dimY * dimX + cy * dimX + cx]) *
			frame1Scalar + static_cast<float>(frame2[dimY * dimX + cy * dimX + cx]) * frame2Scalar) << 8;
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
	double x;
	double y;
	if (cz == 0) {
		x = flowArray[static_cast<unsigned int>(cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)];
		y = flowArray[static_cast<unsigned int>(dimY * dResolutionDivider * dimX * dResolutionDivider) + static_cast<unsigned int>(cy * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)];
	} else {
		x = flowArray[static_cast<unsigned int>(cy * 2 * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)];
		y = flowArray[static_cast<unsigned int>(dimY * dResolutionDivider * dimX * dResolutionDivider) + static_cast<unsigned int>(cy * 2 * dResolutionDivider) * static_cast<unsigned int>(dimX * dResolutionDivider) + static_cast<unsigned int>(cx * dResolutionDivider)];
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

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		p010Array[cy * static_cast<unsigned int>(dimX * dDimScalar) + cx] = static_cast<unsigned short>((fmaxf(fminf(static_cast<float>(0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b), 255.0), 0.0)) * scalar + frame1[cy * dimX + cx] * (1.0 - scalar)) << 8;
	// U/V Channels
	} else if (cz == 1 && cy < (dimY / 2) && cx < dimX) {
		// U Channel
		if (cx % 2 == 0) {
			p010Array[static_cast<unsigned int>(dimY * dimX * dDimScalar) + cy * static_cast<unsigned int>(dimX * dDimScalar) + (cx / 2) * 2] = static_cast<unsigned short>(fmaxf(fminf(static_cast<float>(0.492 * (rgb.b - (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b)) + 128), 255.0), 0.0)) << 8;
		// V Channel
		} else {
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
* Blurs a frame
*
* @param kernelSize: Size of the kernel to use for the blur
* @param directOutput: Whether to output the blurred frame directly
*/
void OpticalFlowCalcSDR::blurFrameArray(const int kernelSize, const bool directOutput) {
	if (!m_bBisNewest) {
		// Launch kernel
		blurFrameKernel << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_blurredFrame1.arrayPtrGPU, kernelSize, m_iDimY, m_iDimX);

		// Convert the NV12 frame to P010 if we are doing direct output
		if (directOutput) {
			convertNV12toP010KernelSDR << <m_grid, m_threads2 >> > (m_blurredFrame1.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		}
	} else {
		// Launch kernel
		blurFrameKernel << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_blurredFrame2.arrayPtrGPU, kernelSize, m_iDimY, m_iDimX);

		// Convert the NV12 frame to P010 if we are doing direct output
		if (directOutput) {
			convertNV12toP010KernelSDR << <m_grid, m_threads2 >> > (m_blurredFrame2.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
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
* Calculates the optical flow between frame1 and frame2
*
* @param iNumIterations: Number of iterations to calculate the optical flow
* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
*/
void OpticalFlowCalcSDR::calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps) {
	// Reset variables
	const int directionIdxOffset = m_iNumLayers * m_iLowDimY * m_iLowDimX; // Offset to index the Y-Offset-Layer
	const int channelIdxOffset = m_iDimY * m_iDimX; // Offset to index the color channel of the current thread
	unsigned int windowDimX = m_iLowDimX;
	unsigned int windowDimY = m_iLowDimY;
	if (iNumIterations == 0 || static_cast<float>(iNumIterations) > ceil(log2f(static_cast<float>(m_iLowDimX)))) {
		iNumIterations = static_cast<unsigned int>(ceil(log2f(static_cast<float>(m_iLowDimX))));
	}

	// Set the starting offset for the current window size
	setInitialOffset << <m_lowGrid, m_threads5 >> > (m_offsetArray12.arrayPtrGPU, m_iNumLayers, m_iLowDimY, m_iLowDimX);

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		// Each step we adjust the offset array to find the ideal offset
		for (unsigned int step = 0; step < iNumSteps; step++) {
			// Reset the summed up delta array
			m_summedUpDeltaArray.zero();

			// 1. Calculate the image deltas with the current offset array
			if (m_bBisNewest) {
				calcImageDeltaSDR << <m_lowGrid, m_threads10 >> > (m_blurredFrame1.arrayPtrGPU, m_blurredFrame2.arrayPtrGPU,
															   m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
															   m_iNumLayers, m_iLowDimY, m_iLowDimX, m_iDimY, m_iDimX,
															   m_dResolutionScalar, directionIdxOffset, channelIdxOffset);
			} else {
				calcImageDeltaSDR << <m_lowGrid, m_threads10 >> > (m_blurredFrame2.arrayPtrGPU, m_blurredFrame1.arrayPtrGPU,
															   m_imageDeltaArray.arrayPtrGPU, m_offsetArray12.arrayPtrGPU,
															   m_iNumLayers, m_iLowDimY, m_iLowDimX, m_iDimY, m_iDimX,
															   m_dResolutionScalar, directionIdxOffset, channelIdxOffset);
			}

			// 2. Sum up the deltas of each window
			calcDeltaSumsSDR << <m_lowGrid, m_threads10 >> > (m_imageDeltaArray.arrayPtrGPU, m_summedUpDeltaArray.arrayPtrGPU,
														  windowDimY, windowDimX, m_iNumLayers, m_iLowDimY, m_iLowDimX);

			// 3. Normalize the summed up delta array and find the best layer
			normalizeDeltaSums << <m_lowGrid, m_threads5 >> > (m_summedUpDeltaArray.arrayPtrGPU, m_lowestLayerArray.arrayPtrGPU,
															   m_offsetArray12.arrayPtrGPU, windowDimY, windowDimX,
															   m_iNumLayers, m_iLowDimY, m_iLowDimX);

			// 4. Adjust the offset array based on the comparison results
			adjustOffsetArray << <m_lowGrid, m_threads1 >> > (m_offsetArray12.arrayPtrGPU, m_lowestLayerArray.arrayPtrGPU,
															  m_statusArray.arrayPtrGPU, windowDimY, windowDimX, 
															  m_iNumLayers, m_iLowDimY, m_iLowDimX);
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
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		// Frame 2 to Frame 1
		} else {
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU, 
																  m_outputFrame.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, scaledDimX, channelIdxOffset);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																	    m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		}
	} else {
		// Frame 1 to Frame 2
		if (bOutput12) {
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																  m_hitCount12.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_outputFrame.arrayPtrGPU, frameScalar12, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, scaledDimX, channelIdxOffset);
			artifactRemovalKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame2.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																		m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_dDimScalar);
		// Frame 2 to Frame 1
		} else {
			warpFrameKernelForOutputSDR << <m_grid, m_threads2 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																  m_hitCount21.arrayPtrGPU, m_ones.arrayPtrGPU,
																  m_outputFrame.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
																  m_iDimY, m_iDimX, m_dResolutionDivider, directionIdxOffset, scaledDimX, channelIdxOffset);
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