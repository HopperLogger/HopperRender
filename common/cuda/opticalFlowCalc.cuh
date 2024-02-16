// opticalFlowCalc.cuh
#pragma once

// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Project Includes
#include <intsafe.h>

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
	*/
	void init(const unsigned int dimY, const unsigned int dimX);

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
    * @param iMaxOffsetDivider: The divider used to calculate the initial global offset
    *
    * @return: The time it took to calculate the optical flow
	*/
	double calculateOpticalFlow(int iNumIterations, int iNumSteps, int iMaxOffsetDivider);

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
	* Downloads the array as a flow image into the output memory pointer
	*
	* @param memPointer: Pointer to the memory to transfer the array to
	* @param saturation: The saturation of the flow image
	* @param value: The value of the flow image
	* @param threshold: The threshold to use for the flow image
	*/
	void downloadFlowAsHSV(unsigned char* memPointer, const double saturation, const double value, const float threshold) const;

	/*
	* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
	*
	* @param memPointer: Pointer to the memory to transfer the array to
	* @param saturation: The saturation of the flow image
	* @param value: The value of the flow image
	* @param threshold: The threshold to use for the flow image
	*/
	void flipFlow();

	// The number of cuda blocks needed
	unsigned int NUM_BLOCKS_X;
	unsigned int NUM_BLOCKS_Y;

	// The number of cuda threads needed
	dim3 grid;
	dim3 threads3;
	dim3 threads2;
	dim3 threads1;

	// Result arrays
	bool bIsInitialized = false; // Whether the optical flow calculation is initialized
	bool bBisNewest = true; // Whether frame1 or frame2 is the newest frame
	GPUArray<unsigned char> frame1; // Array containing the first frame
	GPUArray<unsigned char> frame2; // Array containing the second frame
	GPUArray<unsigned char> imageDeltaArray; // Array containing the absolute difference between the two frames
	GPUArray<int> offsetArray12; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> offsetArray21; // Array containing x,y offsets for each pixel of frame2
	GPUArray<unsigned char> rgboffsetArray; // Array containing the x,y offsets for each pixel of frame1 in rgb format
	GPUArray<int> statusArray; // Array containing the calculation status of each pixel of frame1
	GPUArray<unsigned int> summedUpDeltaArray; // Array containing the summed up delta values of each window
	GPUArray<float> normalizedDeltaArrayA; // Array containing the normalized delta values of each window
	GPUArray<float> normalizedDeltaArrayB; // Array containing the normalized delta values of each window
	GPUArray<bool> isValueDecreasedArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	GPUArray<unsigned char> warpedFrame12; // Array containing the warped frame (frame 1 to frame 2)
	GPUArray<unsigned char> warpedFrame21; // Array containing the warped frame (frame 2 to frame 1)
	GPUArray<unsigned char> blendedFrame; // Array containing the blended frame
	GPUArray<int> hitCount; // Array containing the number of times a pixel was hit
	GPUArray<int> ones; // Array containing only ones for atomic add
	
	// Helper variables
	unsigned int windowDimX; // Initial window size
	unsigned int windowDimY; // Initial window size
	unsigned int currentGlobalOffset; // Initial global offset
};