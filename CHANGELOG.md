## [Version 1.5.6.9] - 2024-03-21 - Fully functional

### Added
- The number of calculation steps in each iteration of the optical flow calculation are now reduced the smaller the window size. This reduces the total number of calculations by a factor of about 3x and reduces artifacts by limiting the amount small windows can move.

### Changed
- Set the minimum number of calculation steps to 3 as it is sufficient
- Set the maximum number of calculation steps to 15 as any more do not provide any benefit and might actually introduce artifacts
- Updated the autoAdjustSettings to increase the calc resolution when the maximum number of steps are reached
- Set the cBuffers from 4 to 5 to reduce playback issues
- Disabled copying of the interlace flag in the output pin to correctly handle DVD playback
- Set the number of planes to 2 to reflect the P010 output format