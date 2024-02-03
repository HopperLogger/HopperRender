// opticalFlowCalc.cuh
#pragma once

// Project Includes
#include "GPUArrayLib.cuh"

#define DEBUG_MODE 0
#define NUM_STEPS 40 // Number of steps executed to find the ideal offset (limits the maximum offset)
#define MAX_OFFSET_DIVIDER 192 // The divider used to calculate the initial global offset

/*
* Calculates the optical flow between frame1 and frame2
*
* @param frame1: The frame to calculate the flow from
* @param frame2: The frame to calculate the flow to
*
* @return: The flow array containing the relative vectors
*/
GPUArray<int> calculateOpticalFlow(GPUArray<unsigned char>& frame1, GPUArray<unsigned char>& frame2);

/*
* Warps frame1 according to the offset array
*
* @param frame1: The frame to warp
* @param offsetArray: The array containing the offsets
*
* @return: The warped frame
*/
GPUArray<unsigned char> warpFrame(GPUArray<unsigned char>& frame1, GPUArray<int>& offsetArray);