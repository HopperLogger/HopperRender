// opticalFlowCalc.cuh
#pragma once

// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Project Includes
#include "GPUArrayLib.cuh"

#define DEBUG_MODE 0
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
	void init(int dimY, int dimX);

	/*
	* Calculates the optical flow between frame1 and frame2
	*
	* @param frame1: The frame to calculate the flow from
	* @param frame2: The frame to calculate the flow to
	*
	* @return: The flow array containing the relative vectors
	*/
	GPUArray<int> calculateOpticalFlow(const GPUArray<unsigned char>& frame1, const GPUArray<unsigned char>& frame2);

	/*
	* Warps frame1 according to the offset array
	*
	* @param frame1: The frame to warp
	* @param offsetArray: The array containing the offsets
	*
	* @return: The warped frame
	*/
	GPUArray<unsigned char> warpFrame(const GPUArray<unsigned char>& frame1, const GPUArray<int>& offsetArray);

private:
	// The number of cuda blocks needed
	int NUM_BLOCKS_X;
	int NUM_BLOCKS_Y;

	// The number of cuda threads needed
	dim3 grid;
	dim3 threads3;
	dim3 threads2;
	dim3 threads1;

	// Result arrays
	GPUArray<unsigned char> imageDeltaArray; // Array containing the absolute difference between the two frames
	GPUArray<int> offsetArray; // Array containing x,y offsets for each pixel of frame1
	GPUArray<int> statusArray; // Array containing the calculation status of each pixel of frame1
	GPUArray<unsigned int> summedUpDeltaArray; // Array containing the summed up delta values of each window
	GPUArray<float> normalizedDeltaArrayA; // Array containing the normalized delta values of each window
	GPUArray<float> normalizedDeltaArrayB; // Array containing the normalized delta values of each window
	GPUArray<bool> isValueDecreasedArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	GPUArray<unsigned char> warpedFrame; // Array containing the warped frame
	GPUArray<int> hitCount; // Array containing the number of times a pixel was hit
	GPUArray<int> ones; // Array containing only ones for atomic add
	GPUArray<unsigned char> layerFrame;
	GPUArray<unsigned char> RGBFrame;
	
	// Helper variables
	int windowDimX; // Initial window size
	int windowDimY; // Initial window size
	int currentGlobalOffset; // Initial global offset
	int numIterations; // Number of iterations needed to get to the smallest window size
};