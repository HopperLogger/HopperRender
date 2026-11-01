#include "opticalFlowCalc.h"
#include <CL/cl.h>
#include <math.h>
#include <exception>
#include <stdexcept>
#include <string>
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
        throw std::runtime_error(std::string("[HopperRender] Unable to create program for kernel ") + kernelFunc + " (OpenCL error " + std::to_string(err) + ")");
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get the build log for debugging
        char buildLog[4096] = "";
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        buildLog[sizeof(buildLog) - 1] = '\0';
        clReleaseProgram(program);
        throw std::runtime_error(std::string("[HopperRender] Unable to build kernel ") + kernelFunc + " (OpenCL error " + std::to_string(err) + "). Build log:\n" + buildLog);
    }

    // Create the compute kernel in the program we wish to run
    *kernel = clCreateKernel(program, kernelFunc, &err);
    clReleaseProgram(program);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("[HopperRender] Unable to create kernel ") + kernelFunc + " (OpenCL error " + std::to_string(err) + ")");
    }
}

// Detects the OpenCL platforms and devices
void OpticalFlowCalc::detectDevices() {
    // Capabilities we are going to check for (we assume HDR as worst case)
    const cl_ulong requiredVRAM = 9.0 * m_frameHeight * m_inputStride + 3.0 * m_frameHeight * m_outputStride +
                                  6lu * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(short) +
                                  MAX_SEARCH_RADIUS * m_opticalFlowFrameHeight * m_opticalFlowFrameWidth * sizeof(unsigned int) +
                                  m_opticalFlowFrameHeight * m_opticalFlowFrameWidth;
    const size_t requiredWorkGroupSizes[3] = {16, 16, 1};
    const cl_ulong requiredSharedMemSize = 2048;

    // Query the available platforms
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        throw std::runtime_error("[HopperRender] No OpenCL platform found (OpenCL error " + std::to_string(err) + ")! Please make sure that the OpenCL drivers for your GPU are installed.");
    }
    cl_platform_id platforms[8];
    if (numPlatforms > 8) numPlatforms = 8;
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    // Collects the reasons why devices were rejected, so we can report them if no device is suitable
    std::string rejectedDevices;

    // Iterate over the available platforms
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        // Query the available devices of this platform
        cl_uint numDevices = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices) != CL_SUCCESS || numDevices == 0) continue;
        if (numDevices > 8) numDevices = 8;

        cl_device_id devices[8];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

        // Iterate over the available devices
        for (cl_uint j = 0; j < numDevices; ++j) {
            // Get the capabilities of the device
            char deviceName[128] = "";
            char deviceVersion[128] = "";
            cl_ulong availableVRAM = 0;
            size_t maxWorkGroupSizes[3] = {0, 0, 0};
            cl_ulong maxSharedMemSize = 0;
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(deviceVersion), deviceVersion, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(availableVRAM), &availableVRAM, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkGroupSizes), maxWorkGroupSizes, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxSharedMemSize), &maxSharedMemSize, NULL);

            // The device version has the format "OpenCL <major>.<minor> <vendor-specific information>"
            int majorVersion = 0, minorVersion = 0;
            sscanf_s(deviceVersion, "OpenCL %d.%d", &majorVersion, &minorVersion);
            const bool supportsRequiredVersion = majorVersion > 1 || (majorVersion == 1 && minorVersion >= 2);
            const bool hasEnoughVRAM = availableVRAM >= requiredVRAM;
            const bool hasEnoughSharedMem = maxSharedMemSize >= requiredSharedMemSize;
            const bool hasEnoughWorkGroupSize = maxWorkGroupSizes[0] >= requiredWorkGroupSizes[0] && maxWorkGroupSizes[1] >= requiredWorkGroupSizes[1] &&
                                                maxWorkGroupSizes[2] >= requiredWorkGroupSizes[2];

            // Check if the device meets the requirements
            if (supportsRequiredVersion && hasEnoughVRAM && hasEnoughSharedMem && hasEnoughWorkGroupSize) {
                printf("[HopperRender] Using %s and %llu MB of VRAM\n", deviceName, requiredVRAM / 1024 / 1024);
                m_clDeviceId = devices[j];
                return;
            }

            // Remember why this device was rejected
            rejectedDevices += std::string("\n  ") + deviceName + " (" + deviceVersion + "):";
            if (!supportsRequiredVersion) {
                rejectedDevices += " OpenCL 1.2 or higher required;";
            }
            if (!hasEnoughVRAM) {
                rejectedDevices += " Not enough VRAM (required: " + std::to_string(requiredVRAM / 1024 / 1024) + " MB, available: " + std::to_string(availableVRAM / 1024 / 1024) + " MB);";
            }
            if (!hasEnoughSharedMem) {
                rejectedDevices += " Not enough shared memory (required: " + std::to_string(requiredSharedMemSize) + " bytes, available: " + std::to_string(maxSharedMemSize) + " bytes);";
            }
            if (!hasEnoughWorkGroupSize) {
                rejectedDevices += " Work group size too small (required: 16x16x1, available: " + std::to_string(maxWorkGroupSizes[0]) + "x" +
                                   std::to_string(maxWorkGroupSizes[1]) + "x" + std::to_string(maxWorkGroupSizes[2]) + ");";
            }
        }
    }

    // No suitable device found
    throw std::runtime_error("[HopperRender] No suitable OpenCL GPU found! Please make sure that your GPU supports OpenCL 1.2 or higher and the OpenCL drivers are installed." +
                             (rejectedDevices.empty() ? std::string(" No OpenCL devices were found.") : " Rejected devices:" + rejectedDevices));
}
