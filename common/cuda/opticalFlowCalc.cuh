#pragma once

#include "device_launch_parameters.h"

#include "GPUArrayLib.cuh"

// Kernel that sets the initial offset array
__global__ void setInitialOffset(int* offsetArray, unsigned int dimZ, unsigned int dimY, unsigned int dimX);

// Kernel that normalizes all the pixel deltas of each window
__global__ void normalizeDeltaSums(const unsigned int* summedUpDeltaArray, unsigned char* globalLowestLayerArray,
	const int* offsetArray, unsigned int windowDim,
	int dimZ, int dimY, int dimX);

// Kernel that adjusts the offset array based on the comparison results
__global__ void adjustOffsetArray(int* offsetArray, const unsigned char* globalLowestLayerArray, unsigned char* statusArray,
	unsigned int windowDim, unsigned int dimZ,
	unsigned int dimY, unsigned int dimX, const bool lastRun);

// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__global__ void flipFlowKernel(const int* flowArray12, int* flowArray21, const unsigned int dimZ,
							   const int dimY, const int dimX, const double dResolutionDivider);

// Kernel that blurs a flow array
__global__ void blurFlowKernel(const int* flowArray, int* blurredFlowArray, int kernelSize, int dimZ, int dimY,
	int dimX, bool offset12);

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
	* @param saturation: The saturation of the flow image
	* @param value: The value of the flow image
	*/
	virtual void drawFlowAsHSV(float saturation, float value) const = 0;

	// The number of cuda threads needed
	dim3 m_lowGrid;
	dim3 m_grid;
	dim3 m_threads10;
	dim3 m_threads5;
	dim3 m_threads2;
	dim3 m_threads1;

	// Variables
	bool m_bBisNewest = true; // Whether frame1 or frame2 is the newest frame
	double m_dResolutionScalar; // Scalar to scale the resolution with
	double m_dResolutionDivider; // Scalar to divide the resolution with
	unsigned int m_iDimX; // Width of the frame
	unsigned int m_iDimY; // Height of the frame
	unsigned int m_iLowDimX; // Width of the frame used by the optical flow calculation
	unsigned int m_iLowDimY; // Height of the frame used by the optical flow calculation
	unsigned int m_iNumLayers; // Number of layers used by the optical flow calculation
	double m_dDimScalar; // Scalar to scale the frame dimensions with depending on the renderer used

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