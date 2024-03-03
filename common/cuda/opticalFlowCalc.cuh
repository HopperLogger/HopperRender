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
	* @param fResolutionDivider: The scalar to divide the resolution with
	*/
	void init(unsigned int dimY, unsigned int dimX, double dDimScalar, float fResolutionDivider);

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
	* @param resolutionScalar: The scalar to scale the resolution with
	*/
	void calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps, float resolutionScalar);

	/*
	* Warps the frames according to the calculated optical flow
	*
	* @param fScalar: The scalar to blend the frames with
	* @param resolutionScalar: The scalar to scale the resolution with
	* @param resolutionDivider: The scalar to divide the resolution with
	*/
	void warpFramesForBlending(float fScalar, float resolutionScalar, float resolutionDivider);

	/*
	* Warps the frames according to the calculated optical flow
	*
	* @param fScalar: The scalar to blend the frames with
	* @param resolutionScalar: The scalar to scale the resolution with
	* @param resolutionDivider: The scalar to divide the resolution with
	* @param bOutput12: Whether to output the warped frame 12 or 21
	* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
	*/
	void warpFramesForOutput(float fScalar, float resolutionScalar, float resolutionDivider, bool bOutput12, double dDimScalar);

	/*
	* Blends warpedFrame1 to warpedFrame2
	*
	* @param dScalar: The scalar to blend the frames with
	* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
	*/
	void blendFrames(float fScalar, double dDimScalar);

	/*
	* Draws the flow as an RGB image
	*
	* @param saturation: The saturation of the flow image
	* @param value: The value of the flow image
	* @param fResolutionDivider: The scalar to divide the resolution with
	* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
	*/
	void drawFlowAsHSV(float saturation, float value, float fResolutionDivider, double dDimScalar) const;

	/*
	* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
	*
	* @param fResolutionDivider: The scalar to divide the resolution with
	*/
	void flipFlow(float fResolutionDivider);

	/*
	* Blurs the offset arrays
	*
	* @param kernelSize: Size of the kernel to use for the blur
	*/
	void blurFlowArrays(int kernelSize);

	/*
	* Converts a frame from NV12 to P010 format (stored in the output frame)
	*
	* @param p010Array: Pointer to the NV12 frame
	* @param dDimScalar: Scalar to scale the frame dimensions with depending on the renderer used
	*/
	void convertNV12toP010(const GPUArray<unsigned char>* nv12Array, double dDimScalar);

	// The number of cuda threads needed
	dim3 grid;
	dim3 highGrid;
	dim3 threads5;
	dim3 threads2;
	dim3 threads1;

	// Variables
	bool m_bIsInitialized = false; // Whether the optical flow calculation is initialized
	bool m_bBisNewest = true; // Whether frame1 or frame2 is the newest frame
	unsigned int m_iWindowDimX; // Current window size of the optical flow calculation
	unsigned int m_iWindowDimY; // Current window size of the optical flow calculation

	// GPU Arrays
	GPUArray<unsigned char> m_frame1; // Array containing the first frame
	GPUArray<unsigned char> m_frame2; // Array containing the second frame
	GPUArray<unsigned char> m_imageDeltaArray; // Array containing the absolute difference between the two frames
	GPUArray<int> m_offsetArray12; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> m_offsetArray21; // Array containing x,y offsets for each pixel of frame2
	GPUArray<int> m_blurredOffsetArray12; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> m_blurredOffsetArray21; // Array containing x,y offsets for each pixel of frame2
	GPUArray<unsigned char> m_statusArray; // Array containing the calculation status of each pixel of frame1
	GPUArray<unsigned int> m_summedUpDeltaArray; // Array containing the summed up delta values of each window
	GPUArray<double> m_normalizedDeltaArray; // Array containing the normalized delta values of each window
	GPUArray<unsigned char> m_lowestLayerArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	GPUArray<unsigned char> m_warpedFrame12; // Array containing the warped frame (frame 1 to frame 2)
	GPUArray<unsigned char> m_warpedFrame21; // Array containing the warped frame (frame 2 to frame 1)
	GPUArray<unsigned short> m_outputFrame; // Array containing the output frame
	GPUArray<int> m_hitCount12; // Array containing the number of times a pixel was hit
	GPUArray<int> m_hitCount21; // Array containing the number of times a pixel was hit
	GPUArray<int> m_ones; // Array containing only ones for atomic add
};