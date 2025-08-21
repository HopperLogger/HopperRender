#pragma once
const char* adjustOffsetArrayKernelHDR = R"CLC(
// Kernel that adjusts the offset array based on the comparison results
__kernel void adjustOffsetArrayKernel(__global short* offsetArray, __global const unsigned char* lowestLayerArray,
                                      const int windowSize, const int searchWindowSize, const int lowDimY,
                                      const int lowDimX, const int step) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);

    if (cy < lowDimY && cx < lowDimX) {
        // We only need the lowestLayer if we are still searching
        const int wx = (cx / windowSize) * windowSize;
        const int wy = (cy / windowSize) * windowSize;
        const unsigned char lowestLayer = lowestLayerArray[wy * lowDimX + wx];
        const short idealRelOffset = (lowestLayer % searchWindowSize) - (searchWindowSize / 2);

        // Calculate the relative offset adjustment that was determined to be ideal
        offsetArray[(step & 1) * lowDimY * lowDimX + cy * lowDimX + cx] += (idealRelOffset * idealRelOffset * (idealRelOffset > 0 ? 1 : -1));
    }
}
)CLC";