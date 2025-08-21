#pragma once
const char* determineLowestLayerKernelSDR = R"CLC(
// Kernel that determines the offset layer with the lowest delta
__kernel void determineLowestLayerKernel(__global unsigned int* summedUpDeltaArray,
                                         __global unsigned char* lowestLayerArray, const int windowSize,
                                         const int searchWindowSize, const int lowDimY, const int lowDimX) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);

    // Find the layer with the lowest value (if we are a window represent)
    if (cy % windowSize == 0 && cx % windowSize == 0) {
        unsigned char lowestLayer = 0;

        for (int z = 1; z < searchWindowSize; ++z) {
            if (summedUpDeltaArray[z * lowDimY * lowDimX + cy * lowDimX + cx] <
                summedUpDeltaArray[lowestLayer * lowDimY * lowDimX + cy * lowDimX + cx]) {
                lowestLayer = z;
            }
        }

        lowestLayerArray[cy * lowDimX + cx] = lowestLayer;
    }
}
)CLC";