#pragma once
const char* copyFrameKernelSDR = R"CLC(
unsigned short apply_levelsY(float value, float black_level, float white_level) {
    return fmax(fmin((value - black_level) / (white_level - black_level) * 65535.0f, 65535.0f), 0.0f);
}

unsigned short apply_levelsUV(float value, float white_level) {
    return fmax(fmin((value - 32768.0f) / white_level * 65535.0f + 32768.0f, 65535.0f), 0.0f);
}

// Kernel that converts a NV12 frame to a P010 frame
__kernel void copyFrameKernel(__global const unsigned char* sourceFrame, __global unsigned short* outputFrame,
                              const int dimY, const int dimX, const int inputStride, const int outputStride, 
                              const float black_level, const float white_level, const int cz) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);

    if (cy >= (dimY >> cz) || cx >= dimX) {
        return;
    }
    
    unsigned short value = sourceFrame[cz * dimY * inputStride + cy * inputStride + cx];
    value = value << 8;
    outputFrame[cz * dimY * outputStride + cy * outputStride + cx] = cz ? apply_levelsUV(value, white_level) : apply_levelsY(value, black_level, white_level);
}
)CLC";