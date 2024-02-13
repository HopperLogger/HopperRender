// opticalFlowCalc.cuh
#pragma once

// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Project Includes
#include <intsafe.h>

#include "GPUArrayLib.cuh"

#define NUM_STEPS 40 // Number of steps executed to find the ideal offset (limits the maximum offset)
#define MAX_OFFSET_DIVIDER 192 // The divider used to calculate the initial global offset

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
	* @param bBisNewest: Whether frame1 or frame2 is the newest frame
	*/
	void calculateOpticalFlow(bool bBisNewest);

	/*
	* Warps frame1 according to the offset array
	*
	* @param dScalar: The scalar to blend the frames with
	* @param bBisNewest: Whether frame1 or frame2 is the newest frame
	*/
	void warpFrame(double dScalar, bool bBisNewest, int offsetX, int offsetY);

	/*
	* Blends frame1 to frame2
	*
	* @param dScalar: The scalar to blend the frames with
	* @param bBisNewest: Whether frame1 or frame2 is the newest frame
	*/
	void blendFrames(double dScalar, bool bBisNewest);

	/*
	* Downloads the array as a flow image into the output memory pointer
	*
	* @param memPointer: Pointer to the memory to transfer the array to
	* @param saturation: The saturation of the flow image
	* @param value: The value of the flow image
	* @param threshold: The threshold to use for the flow image
	*/
	void downloadFlowAsHSV(unsigned char* memPointer, const double saturation, const double value, const float threshold) const;

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
	GPUArray<unsigned char> frame1; // Array containing the first frame
	GPUArray<unsigned char> frame2; // Array containing the second frame
	GPUArray<unsigned char> imageDeltaArray; // Array containing the absolute difference between the two frames
	GPUArray<int> offsetArray; // Array containing x,y offsets for each pixel of frame1
	GPUArray<unsigned char> rgboffsetArray; // Array containing the x,y offsets for each pixel of frame1 in rgb format
	GPUArray<int> statusArray; // Array containing the calculation status of each pixel of frame1
	GPUArray<unsigned int> summedUpDeltaArray; // Array containing the summed up delta values of each window
	GPUArray<float> normalizedDeltaArrayA; // Array containing the normalized delta values of each window
	GPUArray<float> normalizedDeltaArrayB; // Array containing the normalized delta values of each window
	GPUArray<bool> isValueDecreasedArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	GPUArray<unsigned char> warpedFrame; // Array containing the warped frame
	GPUArray<int> hitCount; // Array containing the number of times a pixel was hit
	GPUArray<int> ones; // Array containing only ones for atomic add
	GPUArray<unsigned char> layerFrame;
	GPUArray<unsigned char> layerFrame2;
	
	// Helper variables
	unsigned int windowDimX; // Initial window size
	unsigned int windowDimY; // Initial window size
	unsigned int currentGlobalOffset; // Initial global offset
	unsigned int numIterations; // Number of iterations needed to get to the smallest window size
};