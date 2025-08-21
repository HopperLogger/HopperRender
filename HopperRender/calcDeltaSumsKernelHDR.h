#pragma once
const char* calcDeltaSumsKernelHDR = R"CLC(
#define FIRST_NEIGHBOR_ITERATION 4

// Helper function to get neighbor offset values
inline short getNeighborOffset(__global const short* offsetArray, int neighborIndexX, int neighborIndexY, 
                               int lowDimX, int lowDimY, int directionIndexOffset) {
    return offsetArray[directionIndexOffset + min(max(neighborIndexY, 0), lowDimY - 1) * lowDimX + min(max(neighborIndexX, 0), lowDimX - 1)];
}

// Helper kernel for the calcDeltaSums kernel
void warpReduce8x8(volatile __local unsigned int* partialSums, int tIdx) {
    partialSums[tIdx] += partialSums[tIdx + 32];
    partialSums[tIdx] += partialSums[tIdx + 16];
    partialSums[tIdx] += partialSums[tIdx + 8];
    partialSums[tIdx] += partialSums[tIdx + 4];
    partialSums[tIdx] += partialSums[tIdx + 2];
    partialSums[tIdx] += partialSums[tIdx + 1];
}

// Helper kernel for the calcDeltaSums kernel
void warpReduce4x4(volatile __local unsigned int* partialSums, int tIdx) {
    partialSums[tIdx] += partialSums[tIdx + 16];
    partialSums[tIdx] += partialSums[tIdx + 8];
    partialSums[tIdx] += partialSums[tIdx + 2];
    partialSums[tIdx] += partialSums[tIdx + 1];
}

// Helper kernel for the calcDeltaSums kernel
void warpReduce2x2(volatile __local unsigned int* partialSums, int tIdx) {
    partialSums[tIdx] += partialSums[tIdx + 8];
    partialSums[tIdx] += partialSums[tIdx + 1];
}

// Kernel that sums up all the pixel deltas of each window
__kernel void calcDeltaSumsKernel(__global unsigned int* summedUpDeltaArray, __global const unsigned short* frame1,
                                  __global const unsigned short* frame2, __global const short* offsetArray,
                                  const int dimY, const int dimX, const int lowDimY,
                                  const int lowDimX, const int windowSize, const int searchWindowSize,
                                  const int resolutionScalar, const int iteration, const int step, 
                                  const int deltaScalar, const int neighborBiasScalar) {
    // Shared memory for the partial sums of the current block
    __local unsigned int partialSums[64];

    // Current entry to be computed by the thread
    int cx = get_global_id(0);
    int cy = get_global_id(1);
    const int cz = get_global_id(2);
    const int tIdx = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int scaledCx = cx << resolutionScalar;               // The X-Index of the current thread in the input frames
    int scaledCy = cy << resolutionScalar;               // The Y-Index of the current thread in the input frames
    const int threadIndex2D = cy * lowDimX + cx;         // Standard thread index without Z-Dim
    unsigned int delta = 0;                              // The delta value of the current pixel
    unsigned int offsetBias = 0;                         // The bias of the current offset
    unsigned int neighborBias = 0;                       // The bias of the neighbors (up, down, left, right)
    short neighborOffsetX = 0;                           // The X-Offset of the current neighbor
    short neighborOffsetY = 0;                           // The Y-Offset of the current neighbor
    unsigned short diffToNeighbor = 0;                   // The difference of the current offset to the neighbor's offset

    // Threads that are out of bounds set their delta to 0
    if (cy >= lowDimY || cx >= lowDimX) {
        partialSums[tIdx] = 0;
    } else {
        // Retrieve the offset values for the current thread that are going to be tested
        const short idealOffsetX = offsetArray[threadIndex2D];
        const short idealOffsetY = offsetArray[lowDimY * lowDimX + threadIndex2D];
        short relOffsetAdjustmentX = 0;
        short relOffsetAdjustmentY = 0;
        if (!(step & 1)) {
            relOffsetAdjustmentX = (cz % searchWindowSize) - (searchWindowSize / 2);
            relOffsetAdjustmentX = (relOffsetAdjustmentX * relOffsetAdjustmentX * (relOffsetAdjustmentX > 0 ? 1 : -1));
        } else {
            relOffsetAdjustmentY = (cz % searchWindowSize) - (searchWindowSize / 2);
            relOffsetAdjustmentY = (relOffsetAdjustmentY * relOffsetAdjustmentY * (relOffsetAdjustmentY > 0 ? 1 : -1));
        }
        const short offsetX = idealOffsetX + relOffsetAdjustmentX;
        const short offsetY = idealOffsetY + relOffsetAdjustmentY;
        int newCx = scaledCx + offsetX;
        int newCy = scaledCy + offsetY;

        // Check if we are out of bounds
        if (scaledCx < 0 || scaledCx >= dimX || scaledCy < 0 || scaledCy >= dimY) {
            delta = 0;
        } else {
            // Mirror the projected pixel if it is out of bounds
            if (newCx >= dimX) {
                newCx = dimX - (newCx - dimX + 1);
            } else if (newCx < 0) {
                newCx = -newCx - 1;
            }
            if (newCy >= dimY) {
                newCy = dimY - (newCy - dimY + 1);
            } else if (newCy < 0) {
                newCy = -newCy - 1;
            }

            // Calculate the delta value for the current pixel
            delta = abs_diff(frame1[newCy * dimX + newCx] >> 8, frame2[scaledCy * dimX + scaledCx] >> 8) + 
                    abs_diff(frame1[dimY * dimX + (newCy >> 1) * dimX + (newCx & ~1)] >> 8, frame2[dimY * dimX + (scaledCy >> 1) * dimX + (scaledCx & ~1)] >> 8) + 
                    abs_diff(frame1[dimY * dimX + (newCy >> 1) * dimX + (newCx & ~1) + 1] >> 8, frame2[dimY * dimX + (scaledCy >> 1) * dimX + (scaledCx & ~1) + 1] >> 8);
            delta <<= deltaScalar;
        }

        // Calculate the offset bias
        if (!step) {
            offsetBias = abs(offsetX);
        } else {
            offsetBias = abs(offsetY);
        }

        // Calculate the neighbor biases
        if (iteration >= FIRST_NEIGHBOR_ITERATION) {
            // Relative positions of neighbors
            const int neighborOffsets[4][2] = {
                {0, 2 * windowSize},   // Down
                {2 * windowSize, 0},   // Right
                {-2 * windowSize, 0},  // Left
                {0, -2 * windowSize}  // Up
            };

            // Iterate over neighbors
            for (int i = 0; i < 4; ++i) {
                int neighborIndexX = cx + neighborOffsets[i][0];
                int neighborIndexY = cy + neighborOffsets[i][1];

                // Get the offset values of the current neighbor
                if (!step) {
                    neighborOffsetX = getNeighborOffset(offsetArray, neighborIndexX, neighborIndexY, lowDimX, lowDimY, 0);
                } else {
                    neighborOffsetY = getNeighborOffset(offsetArray, neighborIndexX, neighborIndexY, lowDimX, lowDimY, lowDimY * lowDimX);
                }

                // Calculate the difference between the proposed offset and the neighbor's offset
                if (!step) {
                    diffToNeighbor = abs_diff(neighborOffsetX, offsetX);
                } else {
                    diffToNeighbor = abs_diff(neighborOffsetY, offsetY);
                }

                // Sum differences to the neighbor's offset
                neighborBias += diffToNeighbor;
            }
            neighborBias <<= neighborBiasScalar;
        }
        
        if (windowSize == 1) {
            // Window size of 1x1
            summedUpDeltaArray[cz * lowDimY * lowDimX + cy * lowDimX + cx] = delta + offsetBias + neighborBias;
            return;
        } else {
            // All other window sizes
            partialSums[tIdx] = delta + offsetBias + neighborBias;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over the remaining values
    if (windowSize >= 8) {
        // Window size of 8x8 or larger
        if (tIdx < 32) {
            warpReduce8x8(partialSums, tIdx);
        }
    } else if (windowSize == 4) {
        // Window size of 4x4
        if (get_local_id(1) < 2) {
            // Top 4x4 Blocks
            warpReduce4x4(partialSums, tIdx);
        } else if (get_local_id(1) >= 4 && get_local_id(1) < 6) {
            // Bottom 4x4 Blocks
            warpReduce4x4(partialSums, tIdx);
        }
    } else if (windowSize == 2) {
        // Window size of 2x2
        if ((get_local_id(1) & 1) == 0) {
            warpReduce2x2(partialSums, tIdx);
        }
    }

    // Sync all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sum up the results of all blocks
    if ((cy < lowDimY && cx < lowDimX) &&
        ((windowSize >= 8 && tIdx == 0) || (windowSize == 4 && (tIdx == 0 || tIdx == 4 || tIdx == 32 || tIdx == 36)) ||
        (windowSize == 2 && ((tIdx & 1) == 0 && (get_local_id(1) & 1) == 0)))) {
        const int windowIndexX = cx / windowSize;
        const int windowIndexY = cy / windowSize;
        atomic_add(&summedUpDeltaArray[cz * lowDimY * lowDimX + (windowIndexY * windowSize) * lowDimX + (windowIndexX * windowSize)], partialSums[tIdx]);
    }
}
)CLC";