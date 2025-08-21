#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdbool.h>

#include "config.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define CHECK_ERROR(err)                                                                                  \
    if (err) {                                                                                            \
        throw std::runtime_error(std::string("[HopperRender] OpenCL error occurred in function: ") + __func__ + "\n"); \
    }

class OpticalFlowCalc {
public:
    // Video properties
    int m_frameWidth;          // (stride) width of the frame
    int m_frameHeight;         // Height of the frame
    int m_actualWidth;         // Actual width of the frame (as encoded)
    float m_outputBlackLevel;  // The black level used for the output frame
    float m_outputWhiteLevel;  // The white level used for the output frame

    // Optical flow calculation
    int m_opticalFlowResScalar;     // Determines which resolution scalar will be used for the optical flow calculation
    int m_opticalFlowFrameWidth;    // Width of the frame used by the optical flow calculation
    int m_opticalFlowFrameHeight;   // Height of the frame used by the optical flow calculation
    int m_opticalFlowSearchRadius;  // Search radius used for the optical flow calculation
    double m_ofcCalcTime;           // The time it took to calculate the optical flow
    double m_warpCalcTime;          // The time it took to warp the current intermediate frame
    int m_deltaScalar;              // How many bits the pixel delta values are shifted to the left (affects the weight of the delta values)
    int m_neighborBiasScalar;       // How many bits the neighbor bias values are shifted to the left (affects the weight of the neighbor bias)
    
    // OpenCL variables
    cl_device_id m_clDeviceId;
    cl_context m_clContext;

    // Grids
    size_t m_lowGrid16x16x2[3];
    size_t m_lowGrid16x16x1[3];
    size_t m_lowGrid8x8xL[3];
    size_t m_halfGrid16x16x1[3];
    size_t m_grid16x16x1[3];

    // Threads
    size_t m_threads16x16x1[3];
    size_t m_threads8x8x1[3];

    // Queues
    cl_command_queue m_queue;  // Queue used for the optical flow calculation

    // Events
    cl_event m_ofcStartedEvent;   // Event marking the start of the optical flow calculation
    cl_event m_warpStartedEvent;  // Event marking the start of the interpolation

    // GPU Arrays
    cl_mem m_offsetArray;             // Array containing x,y offsets for each pixel of frame1
    cl_mem m_blurredOffsetArray;      // Array containing x,y offsets for each pixel of frame1
    cl_mem m_summedDeltaValuesArray;  // Array containing the summed up delta values of each window
    cl_mem m_lowestLayerArray;        // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
    cl_mem m_outputFrameArray;        // Array containing the output frame
    cl_mem m_inputFrameArray[2];      // Array containing the last three frames

    // Kernels
    cl_kernel m_calcDeltaSumsKernel;
    cl_kernel m_determineLowestLayerKernel;
    cl_kernel m_adjustOffsetArrayKernel;
    cl_kernel m_blurFlowKernel;
    cl_kernel m_warpFrameKernel;
    cl_kernel m_copyFrameKernel;

    /*
     * Initializes the optical flow calculator
     *
     * @param frameHeight: The height of the video frame
     * @param frameWidth: The width of the video frame
     * @param actualWidth: The encoded width of the video frame
     */
    OpticalFlowCalc() = default;

    /*
     * Frees the memory of the optical flow calculator
     */
    virtual ~OpticalFlowCalc();

    /*
     * Updates the frame arrays and blurs them if necessary
     *
     * @param inputPlanes: Pointer to the input planes of the new source frame
     */
    virtual void updateFrame(unsigned char *inputPlanes) = 0;

    /*
     * Downloads the output frame from the GPU to the CPU
     *
     * @param outputPlanes: Pointer to the output planes where the frame should be stored
     */
    virtual void downloadFrame(unsigned char *outputPlanes) = 0;

    /*
     * Calculates the optical flow between frame1 and frame2
     */
    virtual void calculateOpticalFlow() = 0;

    /*
     * Warps the frames according to the calculated optical flow
     *
     * @param blendingScalar: The scalar to blend the frames with (i.e. the progress between frame1 and frame2)
     * @param frameOutputMode: The mode to output the frames in (0: WarpedFrame12, 1: WarpedFrame21, 2: Both)
     */
    virtual void warpFrames(const float blendingScalar, const int frameOutputMode) = 0;

    /*
     * Converts the NV12 input frame to the P010 output frame
     */
    virtual void copyFrame() = 0;

    void detectDevices();
    void cl_create_kernel(cl_kernel* kernel, cl_context context,
			  cl_device_id deviceId, const char* kernelFunc,
			  const char* kernelSourceFile);
};