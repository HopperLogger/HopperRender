#pragma once

#include "device_launch_parameters.h"

#include "GPUArrayLib.cuh"

// Kernel that converts a BGR frame to a YUV NV12 frame
__global__ void convertBGRtoNV12Kernel(const unsigned char* bgrArray,
				 unsigned char* nv12Array,
				 const unsigned short dimY,
				 const unsigned short dimX);

// Kernel that converts a YUV NV12 frame to a BGR frame
__global__ void convertP010toBGRKernel(const unsigned short* p010Array,
				 unsigned char* bgrArray,
				 const unsigned short dimY,
				 const unsigned short dimX);

// Kernel that blurs a frame
template <typename T> __global__ void blurFrameKernel(const T* frameArray, T* blurredFrameArray,
		const unsigned char kernelSize, const unsigned char chacheSize,
		const unsigned char boundsOffset,
		const unsigned char avgEntriesPerThread,
		const unsigned short remainder, const char lumStart,
		const unsigned char lumEnd, const unsigned short lumPixelCount,
		const char chromStart, const unsigned char chromEnd,
		const unsigned short chromPixelCount, const unsigned short dimY,
		const unsigned short dimX);

// Kernel that sets the initial offset array
__global__ void setInitialOffset(int* offsetArray, const unsigned int numLayers, const unsigned int lowDimY, 
								 const unsigned int lowDimX, const unsigned int layerIdxOffset);

// Kernel that sums up all the pixel deltas of each window
template <typename T>
__global__ void calcDeltaSums(unsigned int* summedUpDeltaArray, const T* frame1, const T* frame2,
							  const int* offsetArray, const unsigned int layerIdxOffset, const unsigned int directionIdxOffset,
						      const unsigned int dimY, const unsigned int dimX, const unsigned int lowDimY, const unsigned int lowDimX,
							  const unsigned int windowDim, const float resolutionScalar);

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

// Kernel that warps a frame according to the offset array
template <typename T, typename S> __global__ void warpFrameKernel(
    const T* frame1, const int* offsetArray, int* hitCount,
    S* warpedFrame, const float frameScalar, const unsigned int lowDimY,
    const unsigned int lowDimX, const unsigned int dimY, const int dimX,
    const float resolutionDivider, const unsigned int directionIdxOffset,
    const unsigned int scaledDimX, const unsigned int channelIdxOffset,
    const unsigned int scaledChannelIdxOffset);

// Kernel that removes artifacts from the warped frame
template <typename T, typename S> __global__ void artifactRemovalKernel(const T* frame1, const int* hitCount, S* warpedFrame,
		      const unsigned int dimY, const unsigned int dimX,
		      const int scaledDimX, const unsigned int channelIdxOffset,
		      const unsigned int scaledChannelIdxOffset);

// Kernel that blends warpedFrame1 to warpedFrame2
template <typename T> __global__ void blendFrameKernel(const T* warpedFrame1, const T* warpedFrame2,
		 unsigned short* outputFrame, const float frame1Scalar,
		 const float frame2Scalar, const unsigned int dimY,
		 const unsigned int dimX, const int scaledDimX,
		 const unsigned int channelIdxOffset);

// Kernel that places half of frame 1 over the outputFrame
template <typename T> __global__ void insertFrameKernel(const T* frame1, unsigned short* outputFrame,
				  const unsigned int dimY,
				  const unsigned int dimX, const int scaledDimX,
				  const unsigned int channelIdxOffset);

// Kernel that places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
template <typename T> __global__ void sideBySideFrameKernel(const T* frame1, const T* warpedFrame1, const T* warpedFrame2, unsigned short* outputFrame, 
									  const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                      const unsigned int dimX, const int scaledDimX, const unsigned int halfDimY, 
									  const unsigned int halfDimX,const unsigned int channelIdxOffset);

// Kernel that creates an HSV flow image from the offset array
template <typename T>
__global__ void convertFlowToHSVKernel(
    const int* flowArray, unsigned short* outputFrame, const T* frame1,
    const float blendScalar, const unsigned int lowDimX,
    const unsigned int dimY, const unsigned int dimX,
    const float resolutionDivider, const unsigned int directionIdxOffset,
    const unsigned int scaledDimX, const unsigned int scaledChannelIdxOffset);

void convertBGRtoNV12(const unsigned char* bgrArray, unsigned char* bgrArrayGPU,
		      unsigned char* nv12Array, unsigned char* nv12ArrayGPU,
		      const unsigned short dimY, const unsigned short dimX);

void convertP010toBGR(const unsigned short* p010ArrayGPU, unsigned char* bgrArray,
		      unsigned char* bgrArrayGPU, const unsigned short dimY,
		      const unsigned short dimX);

class OpticalFlowCalc {
public:
	// Constructor
	OpticalFlowCalc() = default;

	/*
	* Updates the frame arrays and blurs them if necessary
	*
	* @param pInBuffer: Pointer to the input frame
	* @param kernelSize: Size of the kernel to use for the blur
	* @param directOutput: Whether to output the blurred frame directly
	*/
	virtual void updateFrame(unsigned char* pInBuffer, const unsigned char kernelSize, const bool directOutput) = 0;

	/*
	* Copies the frame in the correct format to the output frame
	*
	* @param pInBuffer: Pointer to the input frame
	* @param pOutBuffer: Pointer to the output frame
	*/
	virtual void copyFrame(const unsigned char* pInBuffer, unsigned char* pOutBuffer) = 0;

	/*
	* Copies a frame that is already on the GPU in the correct format to the output buffer
	*
	* @param pOutBuffer: Pointer to the output frame
	* @param exportMode: Whether the input frame is already on the GPU
	*/
	virtual void copyOwnFrame(unsigned char* pOutBuffer, const bool exportMode) = 0;

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
	* @param outputMode: The mode to output the frames in (0: WarpedFrame 1->2, 1: WarpedFrame 2->1, 2: Both for blending)
	*/
	virtual void warpFrames(float fScalar, const int outputMode) = 0;

	/*
	* Blends warpedFrame1 to warpedFrame2
	*
	* @param dScalar: The scalar to blend the frames with
	*/
	virtual void blendFrames(float fScalar) = 0;

	/*
	* Places left half of frame1 over the outputFrame
	*/
	virtual void insertFrame() = 0;

	/*
	* Places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
	* 
	* @param dScalar: The scalar to blend the frames with
	* @param firstFrame: Whether the frame to be placed is the first frame
	*/
	virtual void sideBySideFrame(float fScalar, const bool firstFrame) = 0;

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
	dim3 m_lowGrid8x8x5;
	dim3 m_lowGrid8x8x1;
	dim3 m_grid16x16x1;
	dim3 m_halfGrid16x16x1;
	dim3 m_grid8x8x1;
	
	// Threads
	dim3 m_threads32x32x1;
	dim3 m_threads16x16x2;
	dim3 m_threads16x16x1;
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
	unsigned int m_iSceneChangeThreshold; // The threshold used to determine whether a scene change has occurred
	unsigned int m_iCurrentSceneChange; // How many pixel differences are currently detected

	// GPU Arrays
	GPUArray<int> m_offsetArray12; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> m_offsetArray21; // Array containing x,y offsets for each pixel of frame2
	GPUArray<int> m_blurredOffsetArray12; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> m_blurredOffsetArray21; // Array containing x,y offsets for each pixel of frame2
	GPUArray<unsigned char> m_statusArray; // Array containing the calculation status of each pixel of frame1
	GPUArray<unsigned int> m_summedUpDeltaArray; // Array containing the summed up delta values of each window
	GPUArray<unsigned char> m_lowestLayerArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	GPUArray<unsigned short> m_outputFrame; // Array containing the output frame
	GPUArray<int> m_hitCount12; // Array containing the number of times a pixel was hit
	GPUArray<int> m_hitCount21; // Array containing the number of times a pixel was hit
};