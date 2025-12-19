#define CL_TARGET_OPENCL_VERSION 300

#include "opticalFlowCalcSDR.h"
#include "config.h"
#include <CL/cl.h>
#include <math.h>
#include <exception>
#include <iostream>
#include <stdio.h>
#include <sys/stat.h>

#include "adjustOffsetArrayKernelSDR.h"
#include "blurFlowKernelSDR.h"
#include "calcDeltaSumsKernelSDR.h"
#include "determineLowestLayerKernelSDR.h"
#include "warpFrameKernelSDR.h"
#include "copyFrameKernelSDR.h"

void OpticalFlowCalcSDR::updateFrame(unsigned char* inputPlanes) {
    CHECK_ERROR(clEnqueueWriteBuffer(m_queue, m_inputFrameArray[0], CL_TRUE, 0, m_frameHeight * m_inputStride + (m_frameHeight / 2) * m_inputStride, inputPlanes, 0, NULL, &m_ofcStartedEvent));

    // Swap the frame buffers
    cl_mem temp0 = m_inputFrameArray[0];
    m_inputFrameArray[0] = m_inputFrameArray[1];
    m_inputFrameArray[1] = temp0;
}

void OpticalFlowCalcSDR::downloadFrame(unsigned char* outputPlanes) {
    cl_event warpEndEvent;
    CHECK_ERROR(clEnqueueReadBuffer(m_queue, m_outputFrameArray, CL_TRUE, 0, 2 * (m_frameHeight * m_outputStride + (m_frameHeight / 2) * m_outputStride), outputPlanes, 0, NULL, &warpEndEvent));

    // Evaluate how long the interpolation took
    CHECK_ERROR(clWaitForEvents(1, &m_warpStartedEvent));
    CHECK_ERROR(clWaitForEvents(1, &warpEndEvent));
    cl_ulong start_time, end_time;
    CHECK_ERROR(clGetEventProfilingInfo(m_warpStartedEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &start_time, NULL));
    CHECK_ERROR(clGetEventProfilingInfo(warpEndEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL));
    m_warpCalcTime = (double)(end_time - start_time) / 1e9;
}

void OpticalFlowCalcSDR::calculateOpticalFlow() {
    // Adjust the search radius
    m_lowGrid8x8xL[2] = m_opticalFlowSearchRadius;

    // We set the initial window size to the next larger power of 2
    int windowSize = 1;
    int maxDim = max(m_opticalFlowFrameWidth, m_opticalFlowFrameHeight);
    if (maxDim && !(maxDim & (maxDim - 1))) {
        windowSize = maxDim;
    } else {
        while (maxDim & (maxDim - 1)) {
            maxDim &= (maxDim - 1);
        }
        windowSize = maxDim << 1;
    }
    windowSize /= 2;  // We don't want to compute movement of the entire frame, so we start with smaller windows

    // We only want to compute windows that are 2x2 or larger, so we adjust the needed iterations
    int opticalFlowIterations = NUM_ITERATIONS;
    if (NUM_ITERATIONS == 0 || NUM_ITERATIONS > log2(windowSize)) {
        opticalFlowIterations = log2(windowSize);
    }

    // Prepare the initial offset array
    cl_uint zero = 0;
    CHECK_ERROR(clEnqueueFillBuffer(m_queue, m_offsetArray, &zero, sizeof(short), 0, 2 * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(short), 0, NULL, NULL));

    // We calculate the ideal offset array for each window size
    for (int iter = 0; iter < opticalFlowIterations; iter++) {
        for (int step = 0; step < 2; step++) {
            // Reset the summed up delta array
            CHECK_ERROR(clEnqueueFillBuffer(m_queue, m_summedDeltaValuesArray, &zero, sizeof(unsigned int), 0,
                                            m_opticalFlowSearchRadius * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(unsigned int), 0, NULL, NULL));

            // 1. Calculate the image delta and sum up the deltas of each window
            cl_int err = clSetKernelArg(m_calcDeltaSumsKernel, 1, sizeof(cl_mem), &m_inputFrameArray[0]);
            err |= clSetKernelArg(m_calcDeltaSumsKernel, 2, sizeof(cl_mem), &m_inputFrameArray[1]);
            err |= clSetKernelArg(m_calcDeltaSumsKernel, 9, sizeof(int), &windowSize);
            err |= clSetKernelArg(m_calcDeltaSumsKernel, 10, sizeof(int), &m_opticalFlowSearchRadius);
            err |= clSetKernelArg(m_calcDeltaSumsKernel, 12, sizeof(int), &iter);
            err |= clSetKernelArg(m_calcDeltaSumsKernel, 13, sizeof(int), &step);
            err |= clSetKernelArg(m_calcDeltaSumsKernel, 14, sizeof(int), &m_deltaScalar);
            err |= clSetKernelArg(m_calcDeltaSumsKernel, 15, sizeof(int), &m_neighborBiasScalar);
            CHECK_ERROR(err);
            CHECK_ERROR(clEnqueueNDRangeKernel(m_queue, m_calcDeltaSumsKernel, 3, NULL, m_lowGrid8x8xL, m_threads8x8x1, 0, NULL, NULL));

            // Retrieve the summed up delta for the fully overlapping frames
			if (iter == 0 && step == 0) {
				CHECK_ERROR(clEnqueueReadBuffer(m_queue, m_summedDeltaValuesArray, CL_TRUE, ((m_opticalFlowSearchRadius / 2) - 1) * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(unsigned int), sizeof(unsigned int), &m_totalFrameDelta, 0, NULL, NULL));
                m_totalFrameDelta /= (m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * 10);
            }

            // 2. Find the layer with the lowest delta sum
            CHECK_ERROR(clSetKernelArg(m_determineLowestLayerKernel, 2, sizeof(int), &windowSize));
            CHECK_ERROR(clSetKernelArg(m_determineLowestLayerKernel, 3, sizeof(int), &m_opticalFlowSearchRadius));
            CHECK_ERROR(clEnqueueNDRangeKernel(m_queue, m_determineLowestLayerKernel, 2, NULL, m_lowGrid16x16x1, m_threads16x16x1, 0, NULL, NULL));

            // 3. Adjust the offset array based on the comparison results
            err = clSetKernelArg(m_adjustOffsetArrayKernel, 2, sizeof(int), &windowSize);
            err |= clSetKernelArg(m_adjustOffsetArrayKernel, 3, sizeof(int), &m_opticalFlowSearchRadius);
            err |= clSetKernelArg(m_adjustOffsetArrayKernel, 6, sizeof(int), &step);
            CHECK_ERROR(err);
            CHECK_ERROR(clEnqueueNDRangeKernel(m_queue, m_adjustOffsetArrayKernel, 2, NULL, m_lowGrid16x16x1, m_threads16x16x1, 0, NULL, NULL));
        }

        // 4. Adjust variables for the next iteration
        windowSize = max(windowSize >> 1, (int)1);
    }

    // Blur the flow array
    cl_event ofcEndEvent;
    CHECK_ERROR(clEnqueueNDRangeKernel(m_queue, m_blurFlowKernel, 3, NULL, m_lowGrid16x16x2, m_threads16x16x1, 0, NULL, &ofcEndEvent));

    // Evaluate how long the calculation took
    CHECK_ERROR(clWaitForEvents(1, &m_ofcStartedEvent));
    CHECK_ERROR(clWaitForEvents(1, &ofcEndEvent));
    cl_ulong start_time, end_time;
    CHECK_ERROR(clGetEventProfilingInfo(m_ofcStartedEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &start_time, NULL));
    CHECK_ERROR(clGetEventProfilingInfo(ofcEndEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL));
    m_ofcCalcTime = (double)(end_time - start_time) / 1e9;
    if (m_ofcCalcCount >= CALC_TIME_INTERVAL) { // Update average and peak every 10 seconds (assuming 24 fps)
        m_ofcAvgCalcTime = m_ofcCalcTimeSum / m_ofcCalcCount;
        m_ofcCalcCount = 0;
	    m_ofcCalcTimeSum = 0.0;
	    m_ofcPeakCalcTime = m_ofcCalcTime;
    }
	m_ofcCalcCount++;
    m_ofcCalcTimeSum += m_ofcCalcTime;
    if (m_ofcCalcTime > m_ofcPeakCalcTime) {
        m_ofcPeakCalcTime = m_ofcCalcTime;
    }
}

void OpticalFlowCalcSDR::warpFrames(const float blendingScalar, const int frameOutputMode) {
    // Check if the blending scalar is valid
    if (blendingScalar > 1.0f) {
        printf("Error: Blending scalar is greater than 1.0\n");
        throw std::runtime_error(std::string("[HopperRender] Error in function ") + __func__ + "\n");
    }

    // Calculate the blend scalar
    const float frameScalar12 = blendingScalar;
    const float frameScalar21 = 1.0f - blendingScalar;

    // Warp Frames
    int cz = 0; // Y-Plane
    cl_int err = clSetKernelArg(m_warpFrameKernel, 0, sizeof(cl_mem), &m_inputFrameArray[0]);
    err |= clSetKernelArg(m_warpFrameKernel, 1, sizeof(cl_mem), &m_inputFrameArray[1]);
    err |= clSetKernelArg(m_warpFrameKernel, 4, sizeof(float), &frameScalar12);
    err |= clSetKernelArg(m_warpFrameKernel, 5, sizeof(float), &frameScalar21);
    err |= clSetKernelArg(m_warpFrameKernel, 13, sizeof(int), &frameOutputMode);
    err |= clSetKernelArg(m_warpFrameKernel, 14, sizeof(float), &m_outputBlackLevel);
    err |= clSetKernelArg(m_warpFrameKernel, 15, sizeof(float), &m_outputWhiteLevel);
    err |= clSetKernelArg(m_warpFrameKernel, 16, sizeof(int), &cz);
    CHECK_ERROR(err);
    CHECK_ERROR(clEnqueueNDRangeKernel(m_queue, m_warpFrameKernel, 2, NULL, m_grid16x16x1, m_threads16x16x1, 0, NULL, &m_warpStartedEvent));
    cz = 1; // UV-Plane
    CHECK_ERROR(clSetKernelArg(m_warpFrameKernel, 16, sizeof(int), &cz));
    CHECK_ERROR(clEnqueueNDRangeKernel(m_queue, m_warpFrameKernel, 2, NULL, m_halfGrid16x16x1, m_threads16x16x1, 0, NULL, NULL));
}

void OpticalFlowCalcSDR::copyFrame() {
    // Copy Frame
    int cz = 0; // Y-Plane
    cl_int err = clSetKernelArg(m_copyFrameKernel, 0, sizeof(cl_mem), &m_inputFrameArray[1]);
    err |= clSetKernelArg(m_copyFrameKernel, 6, sizeof(float), &m_outputBlackLevel);
    err |= clSetKernelArg(m_copyFrameKernel, 7, sizeof(float), &m_outputWhiteLevel);
    err |= clSetKernelArg(m_copyFrameKernel, 8, sizeof(int), &cz);
    CHECK_ERROR(err);
    CHECK_ERROR(clEnqueueNDRangeKernel(m_queue, m_copyFrameKernel, 2, NULL, m_grid16x16x1, m_threads16x16x1, 0, NULL, &m_warpStartedEvent));
    cz = 1; // UV-Plane
    CHECK_ERROR(clSetKernelArg(m_copyFrameKernel, 8, sizeof(int), &cz));
    CHECK_ERROR(clEnqueueNDRangeKernel(m_queue, m_copyFrameKernel, 2, NULL, m_halfGrid16x16x1, m_threads16x16x1, 0, NULL, NULL));
}

OpticalFlowCalcSDR::~OpticalFlowCalcSDR() {
    clFinish(m_queue);
    clReleaseMemObject(m_inputFrameArray[0]);
    clReleaseMemObject(m_inputFrameArray[1]);
    clReleaseMemObject(m_outputFrameArray);
    clReleaseMemObject(m_offsetArray);
    clReleaseMemObject(m_blurredOffsetArray);
    clReleaseMemObject(m_summedDeltaValuesArray);
    clReleaseMemObject(m_lowestLayerArray);
    clReleaseKernel(m_calcDeltaSumsKernel);
    clReleaseKernel(m_determineLowestLayerKernel);
    clReleaseKernel(m_adjustOffsetArrayKernel);
    clReleaseKernel(m_blurFlowKernel);
    clReleaseKernel(m_warpFrameKernel);
    clReleaseCommandQueue(m_queue);
    clReleaseContext(m_clContext);
    clReleaseDevice(m_clDeviceId);
}

OpticalFlowCalcSDR::OpticalFlowCalcSDR(const int frameHeight, const int frameWidth,
		     const int inputStride, const int outputStride, int deltaScalar, int neighborScalar,
		    float blackLevel, float whiteLevel, int maxCalcRes) {
    // Set up variables
    m_frameWidth = frameWidth;
    m_frameHeight = frameHeight;
    m_inputStride = inputStride;
    m_outputStride = outputStride;
    m_outputBlackLevel = blackLevel;
    m_outputWhiteLevel = whiteLevel;
    m_opticalFlowSearchRadius = MIN_SEARCH_RADIUS;
    m_opticalFlowResScalar = 0;
    while (frameHeight >> m_opticalFlowResScalar > maxCalcRes) {
	m_opticalFlowResScalar++;
    }
    m_opticalFlowFrameWidth = ceil(m_frameWidth / pow(2, m_opticalFlowResScalar));
    m_opticalFlowFrameHeight = ceil(m_frameHeight / pow(2, m_opticalFlowResScalar));
    m_ofcCalcTime = 0.0;
    m_ofcAvgCalcTime = 0.0;
    m_ofcPeakCalcTime = 0.0;
    m_ofcCalcCount = 0;
    m_ofcCalcTimeSum = 0.0;
    m_warpCalcTime = 0.0;
    m_deltaScalar = deltaScalar;
    m_neighborBiasScalar = neighborScalar;
	m_totalFrameDelta = 0;

    // Define the global and local work sizes
    m_lowGrid16x16x2[0] = ceil(m_opticalFlowFrameWidth / 16.0) * 16.0;
    m_lowGrid16x16x2[1] = ceil(m_opticalFlowFrameHeight / 16.0) * 16.0;
    m_lowGrid16x16x2[2] = 2;
    m_lowGrid16x16x1[0] = ceil(m_opticalFlowFrameWidth / 16.0) * 16.0;
    m_lowGrid16x16x1[1] = ceil(m_opticalFlowFrameHeight / 16.0) * 16.0;
    m_lowGrid16x16x1[2] = 1;
    m_lowGrid8x8xL[0] = ceil(m_opticalFlowFrameWidth / 8.0) * 8.0;
    m_lowGrid8x8xL[1] = ceil(m_opticalFlowFrameHeight / 8.0) * 8.0;
    m_lowGrid8x8xL[2] = m_opticalFlowSearchRadius;
    m_halfGrid16x16x1[0] = ceil(m_frameWidth / 16.0) * 16.0;
    m_halfGrid16x16x1[1] = ceil((m_frameHeight >> 1) / 16.0) * 16.0;
    m_halfGrid16x16x1[2] = 1;
    m_grid16x16x1[0] = ceil(m_frameWidth / 16.0) * 16.0;
    m_grid16x16x1[1] = ceil(m_frameHeight / 16.0) * 16.0;
    m_grid16x16x1[2] = 1;

    m_threads16x16x1[0] = 16;
    m_threads16x16x1[1] = 16;
    m_threads16x16x1[2] = 1;
    m_threads8x8x1[0] = 8;
    m_threads8x8x1[1] = 8;
    m_threads8x8x1[2] = 1;

    // Detect platforms and devices
    detectDevices();

    // Create a context
    cl_int err;
    m_clContext = clCreateContext(0, 1, &m_clDeviceId, NULL, NULL, &err);
    CHECK_ERROR(err);

    // Create the command queues
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    m_queue = clCreateCommandQueueWithProperties(m_clContext, m_clDeviceId, properties, &err);
    CHECK_ERROR(err);

    // Allocate the buffers
    m_inputFrameArray[0] = clCreateBuffer(m_clContext, CL_MEM_READ_ONLY, 1.5 * m_frameHeight * m_inputStride, NULL, &err);
    m_inputFrameArray[1] = clCreateBuffer(m_clContext, CL_MEM_READ_ONLY, 1.5 * m_frameHeight * m_inputStride, NULL, &err);
    m_outputFrameArray = clCreateBuffer(m_clContext, CL_MEM_WRITE_ONLY, 3 * m_frameHeight * m_outputStride, NULL, &err);
    m_offsetArray = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, 2 * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(short), NULL, &err);
    m_blurredOffsetArray = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, 2 * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(short), NULL, &err);
    m_summedDeltaValuesArray = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, MAX_SEARCH_RADIUS * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(unsigned int), NULL, &err);
    m_lowestLayerArray = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, m_opticalFlowFrameHeight * m_opticalFlowFrameWidth, NULL, &err);
    CHECK_ERROR(err);

    // Compile the kernels
    cl_create_kernel(&m_calcDeltaSumsKernel, m_clContext, m_clDeviceId, "calcDeltaSumsKernel", calcDeltaSumsKernelSDR);
    cl_create_kernel(&m_determineLowestLayerKernel, m_clContext, m_clDeviceId, "determineLowestLayerKernel", determineLowestLayerKernelSDR);
    cl_create_kernel(&m_adjustOffsetArrayKernel, m_clContext, m_clDeviceId, "adjustOffsetArrayKernel", adjustOffsetArrayKernelSDR);
    cl_create_kernel(&m_blurFlowKernel, m_clContext, m_clDeviceId, "blurFlowKernel", blurFlowKernelSDR);
    cl_create_kernel(&m_warpFrameKernel, m_clContext, m_clDeviceId, "warpFrameKernel", warpFrameKernelSDR);
    cl_create_kernel(&m_copyFrameKernel, m_clContext, m_clDeviceId, "copyFrameKernel", copyFrameKernelSDR);

    // Set kernel arguments
    err = clSetKernelArg(m_calcDeltaSumsKernel, 0, sizeof(cl_mem), &m_summedDeltaValuesArray);
    err |= clSetKernelArg(m_calcDeltaSumsKernel, 3, sizeof(cl_mem), &m_offsetArray);
    err |= clSetKernelArg(m_calcDeltaSumsKernel, 4, sizeof(int), &m_frameHeight);
    err |= clSetKernelArg(m_calcDeltaSumsKernel, 5, sizeof(int), &m_frameWidth);
    err |= clSetKernelArg(m_calcDeltaSumsKernel, 6, sizeof(int), &m_inputStride);
    err |= clSetKernelArg(m_calcDeltaSumsKernel, 7, sizeof(int), &m_opticalFlowFrameHeight);
    err |= clSetKernelArg(m_calcDeltaSumsKernel, 8, sizeof(int), &m_opticalFlowFrameWidth);
    err |= clSetKernelArg(m_calcDeltaSumsKernel, 11, sizeof(int), &m_opticalFlowResScalar);
    err |= clSetKernelArg(m_determineLowestLayerKernel, 0, sizeof(cl_mem), &m_summedDeltaValuesArray);
    err |= clSetKernelArg(m_determineLowestLayerKernel, 1, sizeof(cl_mem), &m_lowestLayerArray);
    err |= clSetKernelArg(m_determineLowestLayerKernel, 4, sizeof(int), &m_opticalFlowFrameHeight);
    err |= clSetKernelArg(m_determineLowestLayerKernel, 5, sizeof(int), &m_opticalFlowFrameWidth);
    err |= clSetKernelArg(m_adjustOffsetArrayKernel, 0, sizeof(cl_mem), &m_offsetArray);
    err |= clSetKernelArg(m_adjustOffsetArrayKernel, 1, sizeof(cl_mem), &m_lowestLayerArray);
    err |= clSetKernelArg(m_adjustOffsetArrayKernel, 4, sizeof(int), &m_opticalFlowFrameHeight);
    err |= clSetKernelArg(m_adjustOffsetArrayKernel, 5, sizeof(int), &m_opticalFlowFrameWidth);
    err |= clSetKernelArg(m_warpFrameKernel, 2, sizeof(cl_mem), &m_blurredOffsetArray);
    err |= clSetKernelArg(m_warpFrameKernel, 3, sizeof(cl_mem), &m_outputFrameArray);
    err |= clSetKernelArg(m_warpFrameKernel, 6, sizeof(int), &m_opticalFlowFrameHeight);
    err |= clSetKernelArg(m_warpFrameKernel, 7, sizeof(int), &m_opticalFlowFrameWidth);
    err |= clSetKernelArg(m_warpFrameKernel, 8, sizeof(int), &m_frameHeight);
    err |= clSetKernelArg(m_warpFrameKernel, 9, sizeof(int), &m_frameWidth);
    err |= clSetKernelArg(m_warpFrameKernel, 10, sizeof(int), &m_inputStride);
    err |= clSetKernelArg(m_warpFrameKernel, 11, sizeof(int), &m_outputStride);
    err |= clSetKernelArg(m_warpFrameKernel, 12, sizeof(int), &m_opticalFlowResScalar);
    err |= clSetKernelArg(m_blurFlowKernel, 0, sizeof(cl_mem), &m_offsetArray);
    err |= clSetKernelArg(m_blurFlowKernel, 1, sizeof(cl_mem), &m_blurredOffsetArray);
    err |= clSetKernelArg(m_blurFlowKernel, 2, sizeof(int), &m_opticalFlowFrameHeight);
    err |= clSetKernelArg(m_blurFlowKernel, 3, sizeof(int), &m_opticalFlowFrameWidth);
    err |= clSetKernelArg(m_copyFrameKernel, 1, sizeof(cl_mem), &m_outputFrameArray);
    err |= clSetKernelArg(m_copyFrameKernel, 2, sizeof(int), &m_frameHeight);
    err |= clSetKernelArg(m_copyFrameKernel, 3, sizeof(int), &m_frameWidth);
    err |= clSetKernelArg(m_copyFrameKernel, 4, sizeof(int), &m_inputStride);
    err |= clSetKernelArg(m_copyFrameKernel, 5, sizeof(int), &m_outputStride);
    CHECK_ERROR(err);
}
