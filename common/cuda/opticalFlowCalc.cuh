#pragma once

#include "device_launch_parameters.h"

#include "GPUArrayLib.cuh"

class OpticalFlowCalc {
public:
	// Constructor
	OpticalFlowCalc();

	/*
	* Initializes the optical flow calculation
	* 
	* @param dimY: The height of the frame
	* @param dimX: The width of the frame
	* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
	*/
	void init(unsigned int dimY, unsigned int dimX, const double dDimScalar);

	/*
	* Returns whether the optical flow calculation is initialized
	*
	* @return: True if the optical flow calculation is initialized, false otherwise
	*/
	bool isInitialized() const;

	/*
	* Updates the frame1 array
	*/
	void updateFrame1(const BYTE* pInBuffer);

	/*
	* Updates the frame2 array
	*/
	void updateFrame2(const BYTE* pInBuffer);

	/*
	* Calculates the optical flow between frame1 and frame2
	*
	* @param iNumIterations: Number of iterations to calculate the optical flow
	* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
	*/
	void calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps);

	/*
	* Warps frame1 according to the offset array to frame2
	*
	* @param dScalar: The scalar to blend the frames with
	*/
	void warpFrame12(double dScalar);

	/*
	* Warps frame2 according to the offset array to frame1
	*
	* @param dScalar: The scalar to blend the frames with
	*/
	void warpFrame21(double dScalar);

	/*
	* Blends warpedFrame1 to warpedFrame2
	*
	* @param dScalar: The scalar to blend the frames with
	*/
	void blendFrames(double dScalar);

	/*
	* Draws the flow as an RGB image
	*
	* @param saturation: The saturation of the flow image
	* @param value: The value of the flow image
	*/
	void drawFlowAsHSV(double saturation, double value) const;

	/*
	* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
	*/
	void flipFlow();

	/*
	* Blurs the offset arrays
	*
	* @param kernelSize: Size of the kernel to use for the blur
	*/
	void blurFlowArrays(int kernelSize);

	// The number of cuda threads needed
	dim3 grid;
	dim3 threads3;
	dim3 threads2;
	dim3 threads1;

	// Variables
	bool m_bIsInitialized = false; // Whether the optical flow calculation is initialized
	bool m_bBisNewest = true; // Whether frame1 or frame2 is the newest frame
	unsigned int m_iWindowDimX; // Current window size of the optical flow calculation
	unsigned int m_iWindowDimY; // Current window size of the optical flow calculation
	unsigned int m_iCurrentGlobalOffset; // Current global offset of the optical flow calculation

	// GPU Arrays
	GPUArray<unsigned char> m_frame1; // Array containing the first frame
	GPUArray<unsigned char> m_frame2; // Array containing the second frame
	GPUArray<unsigned char> m_imageDeltaArray; // Array containing the absolute difference between the two frames
	GPUArray<int> m_offsetArray12; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> m_offsetArray21; // Array containing x,y offsets for each pixel of frame2
	GPUArray<int> m_blurredOffsetArray12; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> m_blurredOffsetArray21; // Array containing x,y offsets for each pixel of frame2
	GPUArray<int> m_statusArray; // Array containing the calculation status of each pixel of frame1
	GPUArray<unsigned int> m_summedUpDeltaArray; // Array containing the summed up delta values of each window
	GPUArray<float> m_normalizedDeltaArrayA; // Array containing the normalized delta values of each window
	GPUArray<float> m_normalizedDeltaArrayB; // Array containing the normalized delta values of each window
	GPUArray<bool> m_isValueDecreasedArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	GPUArray<unsigned char> m_warpedFrame12; // Array containing the warped frame (frame 1 to frame 2)
	GPUArray<unsigned char> m_warpedFrame21; // Array containing the warped frame (frame 2 to frame 1)
	GPUArray<unsigned char> m_blendedFrame; // Array containing the blended frame
	GPUArray<unsigned short> m_outputFrame; // Array containing the output frame in P010 format
	GPUArray<int> m_hitCount; // Array containing the number of times a pixel was hit
	GPUArray<int> m_ones; // Array containing only ones for atomic add
};