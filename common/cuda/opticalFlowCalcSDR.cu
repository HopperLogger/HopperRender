#include <amvideo.h>
#include <iomanip>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "opticalFlowCalcSDR.cuh"

// Kernel that converts an NV12 array to a P010 array
__global__ void convertNV12toP010KernelSDR(const unsigned char* nv12Array, unsigned short* p010Array, const unsigned int dimY,
										   const unsigned int dimX, const unsigned int scaledDimY, const unsigned int scaledDimX,
										   const unsigned int channelIdxOffset, const unsigned int scaledChannelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Check if the current thread is inside the Y-Channel or the U/V-Channel
	if (cx < scaledDimX && ((cz == 0 && cy < scaledDimY) || (cz == 1 && cy < (scaledDimY >> 1)))) {
		p010Array[cz * scaledChannelIdxOffset + cy * scaledDimX + cx] = static_cast<unsigned short>(nv12Array[cz * channelIdxOffset + cy * dimX + cx]) << 8;
	}
}

/*
* Initializes the SDR optical flow calculator
*
* @param dimY: The height of the frame
* @param dimX: The width of the frame
* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
* @param fResolutionDivider: The scalar to scale the resolution with
*/
OpticalFlowCalcSDR::OpticalFlowCalcSDR(const unsigned int dimY, const unsigned int dimX, const double dDimScalar, const float fResolutionDivider) {
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
	m_lowGrid16x16x5.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / 16.0), 1.0));
	m_lowGrid16x16x5.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / 16.0), 1.0));
	m_lowGrid16x16x5.z = 5;
	m_lowGrid16x16x4.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / 16.0), 1.0));
	m_lowGrid16x16x4.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / 16.0), 1.0));
	m_lowGrid16x16x4.z = 4;
	m_lowGrid16x16x1.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / 16.0), 1.0));
	m_lowGrid16x16x1.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / 16.0), 1.0));
	m_lowGrid16x16x1.z = 1;
	m_lowGrid8x8x5.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / 8.0), 1.0));
	m_lowGrid8x8x5.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / 8.0), 1.0));
	m_lowGrid8x8x5.z = 5;
	m_lowGrid8x8x1.x = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimX) / 8.0), 1.0));
	m_lowGrid8x8x1.y = static_cast<int>(fmax(ceil(static_cast<double>(m_iLowDimY) / 8.0), 1.0));
	m_lowGrid8x8x1.z = 1;
	m_grid16x16x1.x = static_cast<int>(fmax(ceil(dimX / 16.0), 1.0));
	m_grid16x16x1.y = static_cast<int>(fmax(ceil(dimY / 16.0), 1.0));
	m_grid16x16x1.z = 1;
	m_halfGrid16x16x1.x = static_cast<int>(fmax(ceil(dimX / 32.0), 1.0));
	m_halfGrid16x16x1.y = static_cast<int>(fmax(ceil(dimY / 16.0), 1.0));
	m_halfGrid16x16x1.z = 1;
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
	m_threads8x8x1.x = 8;
	m_threads8x8x1.y = 8;
	m_threads8x8x1.z = 1;

	// GPU Arrays
	m_frame1.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_frame2.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_blurredFrame1.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_blurredFrame2.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_offsetArray12.init({2, 5, dimY, dimX});
	m_offsetArray21.init({2, dimY, dimX});
	m_blurredOffsetArray12.init({2, dimY, dimX});
	m_blurredOffsetArray21.init({2, dimY, dimX});
	m_statusArray.init({dimY, dimX});
	m_summedUpDeltaArray.init({5, dimY, dimX});
	m_lowestLayerArray.init({dimY, dimX});
	m_warpedFrame12.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_warpedFrame21.init({1, dimY, dimX}, 0, static_cast<size_t>(1.5 * static_cast<double>(dimY * dimX)));
	m_outputFrame.init({1, dimY, dimX}, 0, static_cast<size_t>(3.0 * dimY * dimX * dDimScalar));
	m_hitCount12.init({1, dimY, dimX});
	m_hitCount21.init({1, dimY, dimX});
}

/*
* Updates the frame arrays and blurs them if necessary
*
* @param pInBuffer: Pointer to the input frame
* @param kernelSize: Size of the kernel to use for the blur
* @param directOutput: Whether to output the blurred frame directly
*/
void OpticalFlowCalcSDR::updateFrame(unsigned char* pInBuffer, const unsigned char kernelSize, const bool directOutput) {
    // We always want frame 2 to be the newest and frame 1 to be the oldest, so we swap the pointers accordingly
	unsigned char* temp;
	if (m_bBisNewest) {
		m_frame1.fillData(pInBuffer);
		blurFrameArray(m_frame1.arrayPtrGPU, m_blurredFrame1.arrayPtrGPU, kernelSize, directOutput);

		// Swap the frame arrays
		temp = m_frame1.arrayPtrGPU;
		m_frame1.arrayPtrGPU = m_frame2.arrayPtrGPU;
		m_frame2.arrayPtrGPU = temp;

		// Swap the blurred frame arrays
		temp = m_blurredFrame1.arrayPtrGPU;
		m_blurredFrame1.arrayPtrGPU = m_blurredFrame2.arrayPtrGPU;
		m_blurredFrame2.arrayPtrGPU = temp;
	} else {
		// Swap the frame arrays
		temp = m_frame1.arrayPtrGPU;
		m_frame1.arrayPtrGPU = m_frame2.arrayPtrGPU;
		m_frame2.arrayPtrGPU = temp;

		// Swap the blurred frame arrays
		temp = m_blurredFrame1.arrayPtrGPU;
		m_blurredFrame1.arrayPtrGPU = m_blurredFrame2.arrayPtrGPU;
		m_blurredFrame2.arrayPtrGPU = temp;

		m_frame2.fillData(pInBuffer);
		blurFrameArray(m_frame2.arrayPtrGPU, m_blurredFrame2.arrayPtrGPU, kernelSize, directOutput);
	}
	m_bBisNewest = !m_bBisNewest;
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
	convertNV12toP010KernelSDR << <m_grid16x16x1, m_threads16x16x2>> > (m_frame1.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iScaledDimY, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);

	// Download the output frame
	m_outputFrame.download(pOutBuffer);
}

/*
* Copies a frame that is already on the GPU in the correct format to the output buffer
*
* @param pOutBuffer: Pointer to the output frame
* @param exportMode: Whether the input frame is already on the GPU
*/
void OpticalFlowCalcSDR::copyOwnFrame(unsigned char* pOutBuffer, const bool exportMode) {
	// Convert the NV12 frame to P010
	convertNV12toP010KernelSDR << <m_grid16x16x1, m_threads16x16x2>> > (m_frame1.arrayPtrGPU, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iScaledDimY, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);

	// Download the output frame
	if (!exportMode) m_outputFrame.download(pOutBuffer);
}

/*
* Blurs a frame
*
* @param frame: Pointer to the frame to blur
* @param blurredFrame: Pointer to the blurred frame
* @param kernelSize: Size of the kernel to use for the blur
* @param directOutput: Whether to output the blurred frame directly
*/
void OpticalFlowCalcSDR::blurFrameArray(const unsigned char* frame, unsigned char* blurredFrame, const unsigned char kernelSize, const bool directOutput) {
	const unsigned char boundsOffset = kernelSize >> 1;
	const unsigned char chacheSize = kernelSize + (boundsOffset << 1);
	const size_t sharedMemSize = chacheSize * chacheSize * sizeof(unsigned int);
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
	const unsigned int NUM_BLOCKS_X = max(static_cast<int>(ceil(static_cast<double>(m_iDimX) / kernelSize)), 1);
	const unsigned int NUM_BLOCKS_Y = max(static_cast<int>(ceil(static_cast<double>(m_iDimY) / kernelSize)), 1);

	// Use dim3 structs for block and grid size
	dim3 gridBF(NUM_BLOCKS_X, NUM_BLOCKS_Y, 2);
	if (!directOutput) gridBF.z = 1; // We only need the luminance channel if we are not doing direct output
	dim3 threadsBF(kernelSize, kernelSize, 1);

	// No need to blur the frame if the kernel size is less than 4
	if (kernelSize < 4) {
		cudaMemcpy(blurredFrame, frame, m_frame2.bytes, cudaMemcpyDeviceToDevice);
	} else {
		// Launch kernel
		blurFrameKernel << <gridBF, threadsBF, sharedMemSize >> > (frame, blurredFrame, kernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, lumStart, lumEnd, lumPixelCount, chromStart, chromEnd, chromPixelCount, m_iDimY, m_iDimX);
	}

	// Convert the NV12 frame to P010 if we are doing direct output
	if (directOutput) {
		convertNV12toP010KernelSDR << <m_grid16x16x1, m_threads16x16x2 >> > (blurredFrame, m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, m_iScaledDimY, m_iScaledDimX, m_iChannelIdxOffset, m_iScaledChannelIdxOffset);
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

	size_t sharedMemSize = 16 * 16 * sizeof(unsigned int);

	// Set layer 0 of the X-Dir to 0
	cudaMemset(m_offsetArray12.arrayPtrGPU, 0, m_iLayerIdxOffset * sizeof(int));
	// Set layers 0-5 of the Y-Dir to 0
	cudaMemset(m_offsetArray12.arrayPtrGPU + m_iDirectionIdxOffset, 0, m_iDirectionIdxOffset * sizeof(int));
	// Set layers 1-4 of the X-Dir to -2,-1,1,2
	setInitialOffset << <m_lowGrid16x16x4, m_threads16x16x1 >> > (m_offsetArray12.arrayPtrGPU, m_iNumLayers, m_iLowDimY, m_iLowDimX, m_iLayerIdxOffset);

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		// Calculate the number of steps for this iteration executed to find the ideal offset (limits the maximum offset)
	    //iNumStepsPerIter = static_cast<unsigned int>(static_cast<double>(iNumSteps) - static_cast<double>(iter) * (static_cast<double>(iNumSteps) / static_cast<double>(iNumIterations)));
		iNumStepsPerIter = max(static_cast<unsigned int>(static_cast<double>(iNumSteps) * exp(-static_cast<double>(3 * iter) / static_cast<double>(iNumIterations))), 1);

		// Each step we adjust the offset array to find the ideal offset
		for (unsigned int step = 0; step < iNumStepsPerIter; step++) {
			// Reset the summed up delta array
			m_summedUpDeltaArray.zero();

			// 1. Calculate the image delta and sum up the deltas of each window
			calcDeltaSums << <iter == 0 ? m_lowGrid16x16x5 : m_lowGrid8x8x5, iter == 0 ? m_threads16x16x1 : m_threads8x8x1, sharedMemSize>> > (m_summedUpDeltaArray.arrayPtrGPU, 
																m_blurredFrame1.arrayPtrGPU,
                                                                m_blurredFrame2.arrayPtrGPU,
															    m_offsetArray12.arrayPtrGPU, m_iLayerIdxOffset, m_iDirectionIdxOffset,
																m_iDimY, m_iDimX, m_iLowDimY, m_iLowDimX, windowDim, m_fResolutionScalar);
			// Check if interpolation is appropriate
			if (iter == 0 && step == 0) {
				cudaMemcpy(&m_iCurrentSceneChange, m_summedUpDeltaArray.arrayPtrGPU, sizeof(unsigned int), cudaMemcpyDeviceToHost);
				m_iCurrentSceneChange /= ((m_iLowDimY * m_iLowDimX) / 30);
				if (m_iCurrentSceneChange == 0 || m_iCurrentSceneChange > m_iSceneChangeThreshold) {
					m_statusArray.zero();
					return;
				}
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

		// 4. Adjust variables for the next iteration
		windowDim = max(windowDim >> 1, 1);
		sharedMemSize = 8 * 8 * sizeof(unsigned int);
		if (windowDim == 1) sharedMemSize = 0;

		// Reset the status array
		m_statusArray.zero();
	}
}

/*
* Warps the frames according to the calculated optical flow
*
* @param fScalar: The scalar to blend the frames with
* @param outputMode: The mode to output the frames in (0: WarpedFrame 1->2, 1: WarpedFrame 2->1, 2: Both for blending)
*/
void OpticalFlowCalcSDR::warpFrames(float fScalar, const int outputMode) {
	// Calculate the blend scalar
	const float frameScalar12 = fScalar;
	const float frameScalar21 = 1.0f - fScalar;

	unsigned int scaledDimX = m_iScaledDimX; 
	unsigned int scaledChannelIdxOffset = m_iScaledChannelIdxOffset;
	if (outputMode > 1) {
		scaledDimX = m_iDimX;
		scaledChannelIdxOffset = m_iChannelIdxOffset;
	}

	// Reset the hit count array
	m_hitCount12.zero();
	m_hitCount21.zero();

	// Create CUDA streams
	cudaStream_t warpStream1, warpStream2;
	cudaStreamCreate(&warpStream1);
	cudaStreamCreate(&warpStream2);

	// Frame 1 to Frame 2
	// Direct Output
	if (outputMode == 0) {
		warpFrameKernel << <m_grid16x16x1, m_threads16x16x2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																	m_hitCount12.arrayPtrGPU,
																	m_outputFrame.arrayPtrGPU, frameScalar12, m_iLowDimY, m_iLowDimX,
																	m_iDimY, m_iDimX, m_fResolutionDivider, m_iLayerIdxOffset, scaledDimX, m_iChannelIdxOffset, scaledChannelIdxOffset);
		artifactRemovalKernel << <m_grid8x8x1, m_threads8x8x2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																						m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, scaledDimX, m_iChannelIdxOffset, scaledChannelIdxOffset);
		cudaStreamSynchronize(warpStream1);
	// Output for blending
	} else {
		warpFrameKernel << <m_grid16x16x1, m_threads16x16x2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_blurredOffsetArray12.arrayPtrGPU,
																	m_hitCount12.arrayPtrGPU,
																	m_warpedFrame12.arrayPtrGPU, frameScalar12, m_iLowDimY, m_iLowDimX,
																	m_iDimY, m_iDimX, m_fResolutionDivider, m_iLayerIdxOffset, scaledDimX, m_iChannelIdxOffset, scaledChannelIdxOffset);
		artifactRemovalKernel << <m_grid8x8x1, m_threads8x8x2, 0, warpStream1 >> > (m_frame1.arrayPtrGPU, m_hitCount12.arrayPtrGPU,
																						m_warpedFrame12.arrayPtrGPU, m_iDimY, m_iDimX, scaledDimX, m_iChannelIdxOffset, scaledChannelIdxOffset);
		cudaStreamSynchronize(warpStream1);
	}

	// Frame 2 to Frame 1
	// Direct Output
	if (outputMode == 1) {
		warpFrameKernel << <m_grid16x16x1, m_threads16x16x2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																	m_hitCount21.arrayPtrGPU,
																	m_outputFrame.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
																	m_iDimY, m_iDimX, m_fResolutionDivider, m_iLayerIdxOffset, scaledDimX, m_iChannelIdxOffset, scaledChannelIdxOffset);
		artifactRemovalKernel << <m_grid8x8x1, m_threads8x8x2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																						m_outputFrame.arrayPtrGPU, m_iDimY, m_iDimX, scaledDimX, m_iChannelIdxOffset, scaledChannelIdxOffset);
		cudaStreamSynchronize(warpStream2);
	// Output for blending
	} else {
		warpFrameKernel << <m_grid16x16x1, m_threads16x16x2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_blurredOffsetArray21.arrayPtrGPU,
																	m_hitCount21.arrayPtrGPU,
																	m_warpedFrame21.arrayPtrGPU, frameScalar21, m_iLowDimY, m_iLowDimX,
																	m_iDimY, m_iDimX, m_fResolutionDivider, m_iLayerIdxOffset, scaledDimX, m_iChannelIdxOffset, scaledChannelIdxOffset);
		artifactRemovalKernel << <m_grid8x8x1, m_threads8x8x2, 0, warpStream2 >> > (m_frame2.arrayPtrGPU, m_hitCount21.arrayPtrGPU,
																						m_warpedFrame21.arrayPtrGPU, m_iDimY, m_iDimX, scaledDimX, m_iChannelIdxOffset, scaledChannelIdxOffset);
		cudaStreamSynchronize(warpStream2);
	}
		 
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
	const float frame1Scalar = 1.0f - fScalar;
	const float frame2Scalar = fScalar;

	// Blend the frames
	blendFrameKernel << <m_grid16x16x1, m_threads16x16x2 >> >(m_warpedFrame12.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU,
												 m_outputFrame.arrayPtrGPU, frame1Scalar, frame2Scalar,
	                                             m_iDimY, m_iDimX, m_iScaledDimX, m_iChannelIdxOffset);
}

/*
* Places left half of frame1 over the outputFrame
*/
void OpticalFlowCalcSDR::insertFrame() {
	insertFrameKernel << <m_halfGrid16x16x1, m_threads16x16x2 >> >(m_frame1.arrayPtrGPU,
												 m_outputFrame.arrayPtrGPU,
	                                             m_iDimY, m_iDimX, m_iScaledDimX, m_iChannelIdxOffset);
}

/*
* Places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
* 
* @param dScalar: The scalar to blend the frames with
* @param firstFrame: Whether the frame to be placed is the first frame
*/
void OpticalFlowCalcSDR::sideBySideFrame(float fScalar, const bool firstFrame) {
	// Calculate the blend scalar
	const float frame1Scalar = 1.0f - fScalar;
	const float frame2Scalar = fScalar;
	const unsigned int halfDimX = m_iDimX >> 1;
	const unsigned int halfDimY = m_iDimY >> 1;

	if (firstFrame) {
		sideBySideFrameKernel << <m_grid16x16x1, m_threads16x16x2 >> >(m_frame2.arrayPtrGPU, m_warpedFrame21.arrayPtrGPU, 
													 m_warpedFrame21.arrayPtrGPU,
													 m_outputFrame.arrayPtrGPU, frame1Scalar, frame2Scalar,
													 m_iDimY, m_iDimX, m_iScaledDimX, halfDimY, halfDimX, m_iChannelIdxOffset);
	} else {
		sideBySideFrameKernel << <m_grid16x16x1, m_threads16x16x2 >> >(m_frame1.arrayPtrGPU, m_warpedFrame12.arrayPtrGPU, 
													 m_warpedFrame21.arrayPtrGPU,
													 m_outputFrame.arrayPtrGPU, frame1Scalar, frame2Scalar,
													 m_iDimY, m_iDimX, m_iScaledDimX, halfDimY, halfDimX, m_iChannelIdxOffset);
	}
}

/*
* Draws the flow as an RGB image
*
* @param blendScalar: The scalar that determines how much of the source frame is blended with the flow
*/
void OpticalFlowCalcSDR::drawFlowAsHSV(const float blendScalar) const {
	convertFlowToHSVKernel << <m_grid16x16x1, m_threads16x16x2 >> > (m_blurredOffsetArray12.arrayPtrGPU, m_outputFrame.arrayPtrGPU,
														m_frame2.arrayPtrGPU, blendScalar, m_iLowDimX, m_iDimY, m_iDimX, 
														m_fResolutionDivider, m_iLayerIdxOffset, m_iScaledDimX, m_iScaledChannelIdxOffset);
}