#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdbool.h>

#include "config.h"
#include "opticalFlowCalc.h"

class OpticalFlowCalcSDR : public OpticalFlowCalc {
public:
    /*
     * Initializes the optical flow calculator
     *
     * @param frameHeight: The height of the video frame
     * @param frameWidth: The width of the video frame
     * @param actualWidth: The encoded width of the video frame
     */
    OpticalFlowCalcSDR(const int frameHeight, const int frameWidth, const int actualWidth);

    /*
     * Frees the memory of the optical flow calculator
     */
    ~OpticalFlowCalcSDR() override;

    /*
     * Updates the frame arrays and blurs them if necessary
     *
     * @param inputPlanes: Pointer to the input planes of the new source frame
     */
    void updateFrame(unsigned char *inputPlanes) override;

    /*
     * Downloads the output frame from the GPU to the CPU
     *
     * @param outputPlanes: Pointer to the output planes where the frame should be stored
     */
    void downloadFrame(unsigned char *outputPlanes) override;

    /*
     * Calculates the optical flow between frame1 and frame2
     */
    void calculateOpticalFlow() override;

    /*
     * Warps the frames according to the calculated optical flow
     *
     * @param blendingScalar: The scalar to blend the frames with (i.e. the progress between frame1 and frame2)
     * @param frameOutputMode: The mode to output the frames in (0: WarpedFrame12, 1: WarpedFrame21, 2: Both)
     */
    void warpFrames(const float blendingScalar, const int frameOutputMode) override;

    /*
     * Converts the NV12 input frame to the P010 output frame
     */
    void copyFrame() override;
};