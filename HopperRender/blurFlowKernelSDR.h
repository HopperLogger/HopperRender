#pragma once
const char* blurFlowKernelSDR = R"CLC(
#define BLOCK_SIZE 16
#define KERNEL_RADIUS 4

// Helper function to mirror the coordinate if it is outside the bounds
int mirrorCoordinate(int pos, int dim) {
    if (pos >= dim) {
        return dim - (pos - dim + 1);
    } else if (pos < 0) {
        return -pos - 1;
    }
    return pos;
}

// Kernel that blurs a flow array
__kernel void blurFlowKernel(__global const short* offsetArray, __global short* blurredOffsetArray, 
                             const int dimY, const int dimX) {
    // Thread and global indices
    const int tx = get_local_id(0);   // Local thread x index
    const int ty = get_local_id(1);   // Local thread y index
    const int gx = get_global_id(0);  // Global x index
    const int gy = get_global_id(1);  // Global y index
    const int gz = get_global_id(2);  // Global z index (layer index)

    // If the kernel radius is 0, just copy the value
    if (KERNEL_RADIUS < 1) {
        blurredOffsetArray[gz * dimX * dimY + gy * dimX + gx] = offsetArray[gz * dimX * dimY + gy * dimX + gx];
        return;
    }

    // Shared memory for the tile, including halos for the blur
    __local short localTile[BLOCK_SIZE + 2 * KERNEL_RADIUS][BLOCK_SIZE + 2 * KERNEL_RADIUS];

    // Dimensions of the local workgroup
    int localSizeX = get_local_size(0);
    int localSizeY = get_local_size(1);

    // Local memory coordinates for this thread
    int lx = tx + KERNEL_RADIUS;
    int ly = ty + KERNEL_RADIUS;

    // Load the main data into shared memory
    localTile[ly][lx] = offsetArray[gz * dimX * dimY + mirrorCoordinate(gy, dimY) * dimX + mirrorCoordinate(gx, dimX)];

    // Load the halo regions
    // Top and bottom halo
    if (ty < KERNEL_RADIUS) {
        int haloYTop = mirrorCoordinate(gy - KERNEL_RADIUS, dimY);
        int haloYBottom = mirrorCoordinate(gy + localSizeY, dimY);
        localTile[ty][lx] = offsetArray[gz * dimX * dimY + haloYTop * dimX + mirrorCoordinate(gx, dimX)];
        localTile[ty + localSizeY + KERNEL_RADIUS][lx] = offsetArray[gz * dimX * dimY + haloYBottom * dimX + mirrorCoordinate(gx, dimX)];
    }

    // Left and right halo
    if (tx < KERNEL_RADIUS) {
        int haloXLeft = mirrorCoordinate(gx - KERNEL_RADIUS, dimX);
        int haloXRight = mirrorCoordinate(gx + localSizeX, dimX);
        localTile[ly][tx] = offsetArray[gz * dimX * dimY + mirrorCoordinate(gy, dimY) * dimX + haloXLeft];
        localTile[ly][tx + localSizeX + KERNEL_RADIUS] = offsetArray[gz * dimX * dimY + mirrorCoordinate(gy, dimY) * dimX + haloXRight];
    }

    // Corner halo
    if (tx < KERNEL_RADIUS && ty < KERNEL_RADIUS) {
        int haloXLeft = mirrorCoordinate(gx - KERNEL_RADIUS, dimX);
        int haloXRight = mirrorCoordinate(gx + localSizeX, dimX);
        int haloYTop = mirrorCoordinate(gy - KERNEL_RADIUS, dimY);
        int haloYBottom = mirrorCoordinate(gy + localSizeY, dimY);
        localTile[ty][tx] = offsetArray[gz * dimX * dimY + haloYTop * dimX + haloXLeft];  // Top Left square
        localTile[ty][tx + localSizeX + KERNEL_RADIUS] = offsetArray[gz * dimX * dimY + haloYTop * dimX + haloXRight];  // Top Right square
        localTile[ty + localSizeY + KERNEL_RADIUS][tx] = offsetArray[gz * dimX * dimY + haloYBottom * dimX + haloXLeft];  // Bottom Left square
        localTile[ty + localSizeY + KERNEL_RADIUS][tx + localSizeX + KERNEL_RADIUS] = offsetArray[gz * dimX * dimY + haloYBottom * dimX + haloXRight];  // Bottom Right square
    }

    // Wait for all threads to finish loading shared memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the blur operation
    if (gx < dimX && gy < dimY) {
        int sum = 0;

        for (int ky = -KERNEL_RADIUS; ky < KERNEL_RADIUS; ++ky) {
            for (int kx = -KERNEL_RADIUS; kx < KERNEL_RADIUS; ++kx) {
                sum += localTile[ly + ky][lx + kx];
            }
        }

        // Average the sum
        int kernelSize = (2 * KERNEL_RADIUS) * (2 * KERNEL_RADIUS);
        blurredOffsetArray[gz * dimX * dimY + gy * dimX + gx] = (short)(sum / kernelSize);
    }
}
)CLC";