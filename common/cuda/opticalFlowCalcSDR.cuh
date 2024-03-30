#pragma once

#include "GPUArrayLib.cuh"
#include "opticalFlowCalc.cuh"

class OpticalFlowCalcSDR : public OpticalFlowCalc {
public:
	/*
	* Initializes the SDR optical flow calculator
	*
	* @param dimY: The height of the frame
	* @param dimX: The width of the frame
	* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
	* @param fResolutionDivider: The scalar to scale the resolution with
	*/
	OpticalFlowCalcSDR(unsigned int dimY, unsigned int dimX, double dDimScalar, float fResolutionDivider);

	/*
	* Updates the frame arrays and blurs them if necessary
	*
	* @param pInBuffer: Pointer to the input frame
	* @param kernelSize: Size of the kernel to use for the blur
	* @param directOutput: Whether to output the blurred frame directly
	*/
	void updateFrame(const unsigned char* pInBuffer, const unsigned char kernelSize, const bool directOutput) override;

	/*
	* Copies the frame in the correct format to the output frame
	*
	* @param pInBuffer: Pointer to the input frame
	* @param pOutBuffer: Pointer to the output frame
	*/
	void copyFrame(const unsigned char* pInBuffer, unsigned char* pOutBuffer) override;

	/*
	* Copies a frame that is already on the GPU in the correct format to the output buffer
	*
	* @param pOutBuffer: Pointer to the output frame
	*/
	void copyOwnFrame(unsigned char* pOutBuffer) override;

	/*
	* Blurs a frame
	*
	* @param frame: Pointer to the frame to blur
	* @param blurredFrame: Pointer to the blurred frame
	* @param kernelSize: Size of the kernel to use for the blur
	* @param directOutput: Whether to output the blurred frame directly
	*/
	void blurFrameArray(const unsigned char* frame, unsigned char* blurredFrame,
						const unsigned char kernelSize, const bool directOutput);

	/*
	* Calculates the optical flow between frame1 and frame2
	*
	* @param iNumIterations: Number of iterations to calculate the optical flow
	* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
	*/
	void calculateOpticalFlow(unsigned int iNumIterations, unsigned int iNumSteps) override;

	/*
	* Warps the frames according to the calculated optical flow
	*
	* @param fScalar: The scalar to blend the frames with
	* @param outputMode: The mode to output the frames in (0: WarpedFrame 1->2, 1: WarpedFrame 2->1, 2: Both for blending)
	*/
	void warpFrames(float fScalar, const int outputMode) override;

	/*
	* Blends warpedFrame1 to warpedFrame2
	*
	* @param dScalar: The scalar to blend the frames with
	*/
	void blendFrames(float fScalar) override;

	/*
	* Places left half of frame1 over the outputFrame
	*/
	void insertFrame() override;

	/*
	* Places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
	* 
	* @param dScalar: The scalar to blend the frames with
	*/
	void sideBySideFrame(float fScalar) override;

	/*
	* Draws the flow as an RGB image
	*
	* @param blendScalar: The scalar that determines how much of the source frame is blended with the flow
	*/
	void drawFlowAsHSV(float blendScalar) const override;

	// GPU Arrays
	GPUArray<unsigned char> m_frame1; // Array containing the first frame
	GPUArray<unsigned char> m_frame2; // Array containing the second frame
	GPUArray<unsigned char> m_blurredFrame1; // Array containing the first frame after blurring
	GPUArray<unsigned char> m_blurredFrame2; // Array containing the second frame after blurring
	GPUArray<unsigned char> m_warpedFrame12; // Array containing the warped frame (frame 1 to frame 2)
	GPUArray<unsigned char> m_warpedFrame21; // Array containing the warped frame (frame 2 to frame 1)
};