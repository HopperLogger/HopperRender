#pragma once
const char* copyFrameKernelHDR = R"CLC(
unsigned short apply_levelsY(float value, float black_level, float white_level) {
    return fmax(fmin((value - black_level) / (white_level - black_level) * 65535.0f, 65535.0f), 0.0f);
}

unsigned short apply_levelsUV(float value, float white_level) {
    return fmax(fmin((value - 32768.0f) / white_level * 65535.0f + 32768.0f, 65535.0f), 0.0f);
}

// Kernel that converts a P010 frame to a P010 frame (accounting for special stride width)
__kernel void copyFrameKernel(__global const unsigned short* sourceFrame, __global unsigned short* outputFrame,
                              const int dimY, const int dimX, const int actualDimX, 
                              const float black_level, const float white_level, const int cz) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);

    if (cy >= (dimY >> cz) || cx >= actualDimX) {
        return;
    }
    
    unsigned short value = sourceFrame[cz * dimY * actualDimX + cy * actualDimX + cx];
    outputFrame[cz * dimY * dimX + cy * dimX + cx] = cz ? apply_levelsUV(value, white_level) : apply_levelsY(value, black_level, white_level);
}
)CLC";