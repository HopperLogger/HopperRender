# HopperRender ChangeLog

## [Version 1.0.0.0] - 2024-01-27 - Initial empty filter
### Added
- Loaded the DirectShow transform filter sdk example and setup of filter settings

## [Version 1.1.0.0] - 2024-02-02 - Initial empty filter

## [Version 1.2.0.0] - 2024-02-08 - Proof of concept
### Added
- Added cuda backend
- Implemented multiple frame output
- Added custom settings and debug message

### Changed
- Improved num int frame calculations
- Code refactoring
- Worked on framerate change
- Continued implementing cuda backend

## [Version 1.3.0.0] - 2024-02-10 - Proof of concept
### Added
- Added features to property panel
- Implemented frame blending

### Changed
- Code refactoring
- Improved frame time detection

## [Version 1.4.0.0] - 2024-02-12 - Proof of concept
### Added
- Implemented flow to hsv function
- Added file attributes

## [Version 1.5.0.0] - 2024-02-14 - Fully functional
### Added
- Implemented actual automatic optical flow calculation

## [Version 1.5.0.1] - 2024-02-14 - Fully functional
### Changed
- Improved frame interpolation

## [Version 1.5.1.0] - 2024-02-16 - Fully functional
### Added
- Added blurring of the optical flow for smoother and more uniform warping
- Fixed frame delivery and implemented frame 2 -> 1 warping

## [Version 1.5.1.1] - 2024-02-17 - Fully functional
### Fixed
- Fixed num samples calculation

## [Version 1.5.1.2] - 2024-02-17 - Fully functional
### Changed
- Updated frame time calculations
- General code clean-up and small bug fixes

## [Version 1.5.2.0] - 2024-02-17 - Fully functional
### Changed
- Interpolation quality improvements
- General code clean-up and small bug fixes

## [Version 1.5.3.0] - 2024-02-19
### Changed
- Changed color format to NV12 and P010 instead of relying on RGB24

## [Version 1.5.3.1] - 2024-02-22 - Fully functional
### Fixed
- Fixed compatibility for madVR and EVR

## [Version 1.5.3.2] - 2024-02-22 - Fully functional
### Changed
- A lot of general bug fixes, simplifications, and code clean-up

## [Version 1.5.3.3] - 2024-02-22 - Fully functional
### Added
- Added manual frame scaling which allows the optical flow calculation to treat the input frames as being lower resolution

## [Version 1.5.4.0] - 2024-02-23
### Changed
- Introduced new optical flow calculation method that uses 5 layers that are calculated simultaneously instead of the step by step approach

## [Version 1.5.4.1] - 2024-02-25 - Fully functional
### Fixed
- Fixed new optical flow calculation

## [Version 1.5.4.2] - 2024-02-25 - Fully functional
### Changed
- Introduced a few performance improvements

## [Version 1.5.4.3] - 2024-02-26 - Only shows bad looking WarpedFrame12 output
### Changed
- General code clean-up and small bug fixes

### Fixed
- Fixed new optical flow calculation

## [Version 1.5.4.4] - 2024-02-26 - Fully functional

## [Version 1.5.4.5] - 2024-02-27 - Fully functional
### Changed
- General code clean-up and small bug fixes

### Fixed
- Fixed 4K playback (i.e. the precision of the render specific dimScalar)

## [Version 1.5.4.6] - 2024-03-03 - Fully functional
### Fixed
- Fixed low res optical flow calculation

## [Version 1.5.4.7] - 2024-03-03 - Fully functional
### Changed
- General code clean-up and small bug fixes

## [Version 1.5.5.0] - 2024-03-06 - Fully functional
### Added
- Added support for P010 input i.e. HDR video (the metadata is not passed along however, so the colors look wrong)
- Added automatic frame scaling of the optical flow calculation that treats the input frames as being lower resolution

### Changed
- Configured release build settings to compile a more optimized binary

## [Version 1.5.5.1] - 2024-03-07 - Only shows HSV flow
### Fixed
- Several bug fixes regarding the HSV flow output

## [Version 1.5.5.2] - 2024-03-07 - Fully functional

## [Version 1.5.5.3] - 2024-03-07 - Fully functional

## [Version 1.5.5.4] - 2024-03-07 - Not working
### Added
- Added automatic saving and loading of changes made in the setting panel from the registry

## [Version 1.5.6.0] - 2024-03-08 - Not working

## [Version 1.5.6.1] - 2024-03-09 - Not working
### Fixed
- Fixed color channel consideration in the optical flow calculation

## [Version 1.5.6.2] - 2024-03-10 - Not working
### Changed
- General code cleanup regarding variable names and extracting long formulas into separate variables

## [Version 1.5.6.3] - 2024-03-11 - Fully functional
### Changed
- Improved quality by blurring frames before optical flow calculation and added a setting for the frame blur kernel size in the setting page as well as the option to output the blurred frame

## [Version 1.5.6.4] - 2024-03-12 - Fully functional
### Changed
- Introduced proper sum reduction kernel to significantly speed up the calcDeltaSum kernel (in the first iteration this provides a speed up of up to 87x!)

## [Version 1.5.6.5] - 2024-03-17 - Fully functional
### Changed
- Improved blurFrameKernel performance by using shared memory to cache all pixels used by each threadblock

## [Version 1.5.6.6] - 2024-03-17 - Fully functional
### Changed
- The offset array is now correctly adjusted in the last calculation step for the next iteration
- The calcDeltaSum kernel now correctly reduces the imageDeltaArray at window sizes below 4x4

## [Version 1.5.6.7] - 2024-03-18 - Fully functional
### Added
- Test Mode flag which disables the autoAdjustSettings feature and sets the number of calculation steps to 1 which allows consistent performance measurements

### Changed
- Significantly improved the performance of the calcImageDelta kernel (about 2.7x) by using (among other optimizations) float instead of double precision as well as a larger threadblock size

## [Version 1.5.6.8] - 2024-03-19 - Fully functional
### Changed
- Combined SDR/HDR specific kernels to type template kernels
- General code cleanup regarding variable names and extracting long formulas into separate variables
- Improved performance by extracting static formulas out of the kernels, performing bit operations instead of division/modulo/multiplication/etc., and some usage of float instead of double
- The adjustOffsetArray kernel is now about 4x as fast as before

### Fixed
- When preparing the offset array for the next iteration, the current X offset was written in the wrong layer

### Removed
- Removed shared memory caching from the adjustOffsetArray kernel as it didn't provide any speed-up

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