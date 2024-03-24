#pragma once

#include "device_launch_parameters.h"

#include "GPUArrayLib.cuh"

// Kernel that sets the initial offset array
__global__ void setInitialOffset(int* offsetArray, const unsigned int numLayers, const unsigned int lowDimY, 
								 const unsigned int lowDimX, const unsigned int layerIdxOffset);

// Kernel that calculates the absolute difference between two frames using the offset array
template <typename T>
__global__ void calcImageDelta(const T* frame1, const T* frame2, T* imageDeltaArray,
	const int* offsetArray, const unsigned int lowDimY, const unsigned int lowDimX,
	const unsigned int dimY, const unsigned int dimX, const float resolutionScalar, const unsigned int directionIdxOffset,
	const unsigned int channelIdxOffset);

// Kernel that sums up all the pixel deltas of each window for window sizes of at least 8x8
template <typename T>
__global__ void calcDeltaSums8x8(const T* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);

// Kernel that sums up all the pixel deltas of each window for window sizes of 4x4
template <typename T>
__global__ void calcDeltaSums4x4(const T* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);

// Kernel that sums up all the pixel deltas of each window for window sizes of 2x2
template <typename T>
__global__ void calcDeltaSums2x2(const T* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);

// Kernel that sums up all the pixel deltas of each window for window sizes of 1x1
template <typename T>
__global__ void calcDeltaSums1x1(const T* imageDeltaArray, unsigned int* summedUpDeltaArray, const unsigned int layerIdxOffset,
	const unsigned int channelIdxOffset, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int windowDim);

// Kernel that normalizes all the pixel deltas of each window
__global__ void normalizeDeltaSums(const unsigned int* summedUpDeltaArray, unsigned char* globalLowestLayerArray,
	const int* offsetArray, const unsigned int windowDim, int numPixels,
	const unsigned int directionIdxOffset, const unsigned int layerIdxOffset,
	const unsigned int numLayers, const unsigned int lowDimY, const unsigned int lowDimX);

// Kernel that adjusts the offset array based on the comparison results
__global__ void adjustOffsetArray(int* offsetArray, const unsigned char* globalLowestLayerArray, unsigned char* statusArray,
	unsigned int windowDim, const unsigned int directionIdxOffset, const unsigned int layerIdxOffset, unsigned int numLayers, 
	unsigned int lowDimY, unsigned int lowDimX, const bool lastRun);

// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__global__ void flipFlowKernel(const int* flowArray12, int* flowArray21, const int lowDimY, const int lowDimX, 
							   const float resolutionDivider, const unsigned int directionIdxOffset,
							   const unsigned int layerIdxOffset);

// Kernel that blurs a flow array
__global__ void blurFlowKernel(const int* flowArray, int* blurredFlowArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char start,
								const unsigned char end, const unsigned short pixelCount, const unsigned short numLayers,
								const unsigned short lowDimY, const unsigned short lowDimX);

// Kernel that removes artifacts from the warped frame
template <typename T>
__global__ void artifactRemovalKernelForBlending(const T* frame1, const int* hitCount, T* warpedFrame,
	const unsigned int dimY, const unsigned int dimX, const unsigned int channelIdxOffset);

class OpticalFlowCalc {
public:
	// Constructor
	OpticalFlowCalc() = default;

	/*
	* Updates the frame1 array
	*
	* @param pInBuffer: Pointer to the input frame
	*/
	virtual void updateFrame1(const unsigned char* pInBuffer) = 0;

	/*
	* Updates the frame2 array
	*
	* @param pInBuffer: Pointer to the input frame
	*/
	virtual void updateFrame2(const unsigned char* pInBuffer) = 0;

	/*
	* Copies the frame in the correct format to the output frame
	*
	* @param pInBuffer: Pointer to the input frame
	* @param pOutBuffer: Pointer to the output frame
	*/
	virtual void copyFrame(const unsigned char* pInBuffer, unsigned char* pOutBuffer) = 0;

	/*
	* Blurs a frame
	*
	* @param kernelSize: Size of the kernel to use for the blur
	* @param directOutput: Whether to output the blurred frame directly
	*/
	virtual void blurFrameArray(const unsigned char kernelSize, const bool directOutput) = 0;

	/*
	* Calculates the optical flow between frame1 and frame2
	*
	* @param iNumIterations: Number of iterations to calculate the optical flow
	* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
	*/
	virtual void calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps) = 0;

	/*
	* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
	*/
	void flipFlow() const;

	/*
	* Blurs the offset arrays
	*
	* @param kernelSize: Size of the kernel to use for the blur
	*/
	void blurFlowArrays(int kernelSize) const;

	/*
	* Warps the frames according to the calculated optical flow
	*
	* @param fScalar: The scalar to blend the frames with
	* @param bOutput12: Whether to output the warped frame 12 or 21
	*/
	virtual void warpFramesForOutput(float fScalar, bool bOutput12) = 0;

	/*
	* Warps the frames according to the calculated optical flow
	*
	* @param fScalar: The scalar to blend the frames with
	*/
	virtual void warpFramesForBlending(float fScalar) = 0;

	/*
	* Blends warpedFrame1 to warpedFrame2
	*
	* @param dScalar: The scalar to blend the frames with
	*/
	virtual void blendFrames(float fScalar) = 0;

	/*
	* Draws the flow as an RGB image
	*
	* @param blendScalar: The scalar that determines how much of the source frame is blended with the flow
	*/
	virtual void drawFlowAsHSV(const float blendScalar) const = 0;

	// Grids
	dim3 m_lowGrid32x32x1;
	dim3 m_lowGrid16x16x5;
	dim3 m_lowGrid16x16x4;
	dim3 m_lowGrid16x16x1;
	dim3 m_lowGrid8x8x1;
	dim3 m_grid16x16x1;
	dim3 m_grid8x8x1;
	
	// Threads
	dim3 m_threads32x32x1;
	dim3 m_threads16x16x2;
	dim3 m_threads16x16x1;
	dim3 m_threads8x8x10;
	dim3 m_threads8x8x5;
	dim3 m_threads8x8x2;
	dim3 m_threads8x8x1;

	// Variables
	bool m_bBisNewest = true; // Whether frame1 or frame2 is the newest frame
	float m_fResolutionScalar; // Scalar to scale the resolution with
	float m_fResolutionDivider; // Scalar to divide the resolution with
	unsigned int m_iDimX; // Width of the frame
	unsigned int m_iDimY; // Height of the frame
	unsigned int m_iLowDimX; // Width of the frame used by the optical flow calculation
	unsigned int m_iLowDimY; // Height of the frame used by the optical flow calculation
	unsigned int m_iNumLayers; // Number of layers used by the optical flow calculation
	double m_dDimScalar; // Scalar to scale the frame dimensions with depending on the renderer used
	unsigned int m_iDirectionIdxOffset; // m_iNumLayers * m_iLowDimY * m_iLowDimX
	unsigned int m_iLayerIdxOffset; // m_iLowDimY * m_iLowDimX
	unsigned int m_iChannelIdxOffset; // m_iDimY * m_iDimX
	unsigned int m_iScaledChannelIdxOffset; // m_iDimY * m_iDimX * m_dDimScalar
	unsigned int m_iScaledDimX; // m_iDimX * m_dDimScalar;
	unsigned int m_iScaledDimY; // m_iDimY * m_dDimScalar;

	// GPU Arrays
	GPUArray<int> m_offsetArray12; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> m_offsetArray21; // Array containing x,y offsets for each pixel of frame2
	GPUArray<int> m_blurredOffsetArray12; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> m_blurredOffsetArray21; // Array containing x,y offsets for each pixel of frame2
	GPUArray<unsigned char> m_statusArray; // Array containing the calculation status of each pixel of frame1
	GPUArray<unsigned int> m_summedUpDeltaArray; // Array containing the summed up delta values of each window
	GPUArray<double> m_normalizedDeltaArray; // Array containing the normalized delta values of each window
	GPUArray<unsigned char> m_lowestLayerArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	GPUArray<unsigned short> m_outputFrame; // Array containing the output frame
	GPUArray<int> m_hitCount12; // Array containing the number of times a pixel was hit
	GPUArray<int> m_hitCount21; // Array containing the number of times a pixel was hit
	GPUArray<int> m_ones; // Array containing only ones for atomic add
};