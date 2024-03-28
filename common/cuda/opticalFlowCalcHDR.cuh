#pragma once

#include "GPUArrayLib.cuh"
#include "opticalFlowCalc.cuh"

class OpticalFlowCalcHDR : public OpticalFlowCalc {
public:
	/*
	* Initializes the HDR optical flow calculator
	*
	* @param dimY: The height of the frame
	* @param dimX: The width of the frame
	* @param dDimScalar: The scalar to scale the frame dimensions with depending on the renderer used
	* @param fResolutionDivider: The scalar to scale the resolution with
	*/
	OpticalFlowCalcHDR(unsigned int dimY, unsigned int dimX, double dDimScalar, float fResolutionDivider);

	/*
	* Updates the frame1 array
	*
	* @param pInBuffer: Pointer to the input frame
	*/
	void updateFrame1(const unsigned char* pInBuffer) override;

	/*
	* Updates the frame2 array
	*
	* @param pInBuffer: Pointer to the input frame
	*/
	void updateFrame2(const unsigned char* pInBuffer) override;

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
	* @param bUseFrame2: Whether to use frame2 or frame1
	* @param pOutBuffer: Pointer to the output frame
	*/
	void copyOwnFrame(const bool bUseFrame2, unsigned char* pOutBuffer) override;

	/*
	* Blurs a frame
	*
	* @param kernelSize: Size of the kernel to use for the blur
	* @param directOutput: Whether to output the blurred frame directly
	*/
	void blurFrameArray(const unsigned char kernelSize, const bool directOutput) override;

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
	* @param bOutput12: Whether to output the warped frame 12 or 21
	*/
	void warpFramesForOutput(float fScalar, bool bOutput12) override;

	/*
	* Warps the frames according to the calculated optical flow
	*
	* @param fScalar: The scalar to blend the frames with
	*/
	void warpFramesForBlending(float fScalar) override;

	/*
	* Blends warpedFrame1 to warpedFrame2
	*
	* @param dScalar: The scalar to blend the frames with
	*/
	void blendFrames(float fScalar) override;

	/*
	* Draws the flow as an RGB image
	*
	* @param blendScalar: The scalar that determines how much of the source frame is blended with the flow
	*/
	void drawFlowAsHSV(float blendScalar) const override;

	// GPU Arrays
	GPUArray<unsigned short> m_frame1; // Array containing the first frame
	GPUArray<unsigned short> m_frame2; // Array containing the second frame
	GPUArray<unsigned short> m_blurredFrame1; // Array containing the first frame after blurring
	GPUArray<unsigned short> m_blurredFrame2; // Array containing the second frame after blurring
	GPUArray<unsigned short> m_warpedFrame12; // Array containing the warped frame (frame 1 to frame 2)
	GPUArray<unsigned short> m_warpedFrame21; // Array containing the warped frame (frame 2 to frame 1)
};