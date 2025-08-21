#include "opticalFlowCalc.h"
#include <CL/cl.h>
#include <math.h>
#include <exception>
#include <iostream>
#include <stdio.h>
#include <sys/stat.h>

OpticalFlowCalc::~OpticalFlowCalc() {

}

// Function to create an OpenCL kernel
void OpticalFlowCalc::cl_create_kernel(cl_kernel* kernel, cl_context context, cl_device_id deviceId, const char* kernelFunc, const char* kernelSourceFile) {
    cl_int err;

    // Create the compute program from the source buffer
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceFile, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create program\n");
        throw std::runtime_error(std::string("[HopperRender] Error in function ") + __func__ + "\n");
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get and print the build log for debugging
        char buildLog[4096];
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        fprintf(stderr, "Build log: %s\n", buildLog);
        throw std::runtime_error(std::string("[HopperRender] Error in function ") + __func__ + "\n");
    }

    // Create the compute kernel in the program we wish to run
    *kernel = clCreateKernel(program, kernelFunc, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create kernel %s\n", kernelFunc);
        throw std::runtime_error(std::string("[HopperRender] Error in function ") + __func__ + "\n");
    }

    clReleaseProgram(program);
}

// Detects the OpenCL platforms and devices
void OpticalFlowCalc::detectDevices() {
    // Capabilities we are going to check for
    cl_ulong availableVRAM;
    const cl_ulong requiredVRAM = 6.0 * m_frameHeight * m_frameWidth +
                                  4lu * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(short) +
                                  MAX_SEARCH_RADIUS * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(unsigned int) +
                                  m_opticalFlowFrameHeight * m_opticalFlowFrameWidth;
    size_t maxWorkGroupSizes[3];
    const size_t requiredWorkGroupSizes[3] = {16, 16, 1};
    cl_ulong maxSharedMemSize;
    const cl_ulong requiredSharedMemSize = 2048;

    // Query the available platforms
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        printf("Error getting platform count: %d\n", err);
        throw std::runtime_error(std::string("[HopperRender] Error in function ") + __func__ + "\n");
    }
    cl_platform_id platforms[8];
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    // Iterate over the available platforms
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        // Query the available devices of this platform
        cl_uint numDevices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

        if (numDevices == 0) continue;

        cl_device_id devices[8];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

        // Iterate over the available devices
        for (cl_uint j = 0; j < numDevices; ++j) {
            // Get the capabilities of the device
            char deviceName[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(availableVRAM), &availableVRAM, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkGroupSizes), maxWorkGroupSizes, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxSharedMemSize), &maxSharedMemSize, NULL);

            // Check if the device meets the requirements
            if (availableVRAM >= requiredVRAM && maxSharedMemSize >= requiredSharedMemSize && maxWorkGroupSizes[0] >= requiredWorkGroupSizes[0] && maxWorkGroupSizes[1] >= requiredWorkGroupSizes[1] &&
                maxWorkGroupSizes[2] >= requiredWorkGroupSizes[2]) {
                printf("[HopperRender] Using %s and %lu MB of VRAM\n", deviceName, requiredVRAM / 1024 / 1024);
                m_clDeviceId = devices[j];
		        return;
            }
        }
    }

    // No suitable device found
    printf("Error: No suitable OpenCL GPU found! Please make sure that your GPU supports OpenCL 1.2 or higher and the OpenCL drivers are installed.\n");
    if (availableVRAM < requiredVRAM) {
        printf("Error: Not enough VRAM available! Required: %lu MB, Available: %lu MB\n", requiredVRAM / 1024 / 1024, availableVRAM / 1024 / 1024);
    }
    if (maxSharedMemSize < requiredSharedMemSize) {
        printf("Error: Not enough shared memory available! Required: %lu bytes, Available: %lu bytes\n", requiredSharedMemSize, maxSharedMemSize);
    }
    if (maxWorkGroupSizes[0] < requiredWorkGroupSizes[0] || maxWorkGroupSizes[1] < requiredWorkGroupSizes[1] || maxWorkGroupSizes[2] < requiredWorkGroupSizes[2]) {
        printf("Error: Not enough work group sizes available! Required: %lu, %lu, %lu, Available: %lu, %lu, %lu\n", requiredWorkGroupSizes[0], requiredWorkGroupSizes[1], requiredWorkGroupSizes[2],
               maxWorkGroupSizes[0], maxWorkGroupSizes[1], maxWorkGroupSizes[2]);
    }
    throw std::runtime_error(std::string("[HopperRender] Error in function ") + __func__ + "\n");
}