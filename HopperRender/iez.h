#ifndef __IEZ__
#define __IEZ__

#ifdef __cplusplus
extern "C" {
#endif

// { fd5010a3-8ebe-11ce-8183-00aa00577da1 }
DEFINE_GUID(IID_SettingsInterface,
            0xfd5010a3, 0x8ebe, 0x11ce, 0x81, 0x83, 0x00, 0xaa, 0x00, 0x57, 0x7d, 0xa1);

DECLARE_INTERFACE_(SettingsInterface, IUnknown) {
	STDMETHOD(GetCurrentSettings)(THIS_
		bool* pbActivated, // Whether the filter is activated by the user
		int* piFrameOutput, // What frame output to use (0: WarpedFrame 1 -> 2, 1: WarpedFrame 2 -> 1, 2: BlendedFrame, 3: HSV Flow, 4: Blurred Frames, 5: Side-by-side 1, 6: Side-by-side 2, 7: BlendedFrame w/o warping)
		double* pdTargetFPS, // The effective target frames per second
		int* piFrameRateMode, // How the target FPS is determined (-1: Half the display refresh rate, 0: Display refresh rate, 1: Custom target FPS, >=2: integer multiplier of the source FPS)
		int* piDeltaScalar,
		int* piNeighborScalar,
		int* piBlackLevel,
		int* piWhiteLevel,
		int* piSceneChangeThreshold,
		int* piIntActiveState, // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)
		double* pdSourceFPS, // The source frames per second
		double* pdOFCCalcTime,
		double* pdAVGOFCCalcTime,
		double* pdPeakOFCCalcTime,
		double* pdWarpCalcTime,
		int* piDimX, // The width of the frame
		int* piDimY, // The height of the frame
		int* piLowDimX, // The width of the downscaled frame used in the optical flow calculation
		int* piLowDimY, // The height of the downscaled frame used in the optical flow calculation
		unsigned int* piTotalFrameDelta, // The peak scene change delta1 from the last second
		unsigned int* piTotalFrameDelta2, // The corresponding scene change delta2 at the peak
		unsigned int* piBufferFrames, // Number of additional frames to buffer at the start
		int* piSearchRadius // The optical flow search radius
	) PURE;

	STDMETHOD(UpdateUserSettings)(THIS_
		bool bActivated, // Whether the filter is activated by the user
		int iFrameOutput, // What frame output to use (0: WarpedFrame 1 -> 2, 1: WarpedFrame 2 -> 1, 2: BlendedFrame, 3: HSV Flow, 4: Blurred Frames, 5: Side-by-side 1, 6: Side-by-side 2, 7: BlendedFrame w/o warping)
		double dTargetFPS, // The custom target frames per second (used when iFrameRateMode is 1)
		int iFrameRateMode, // How the target FPS is determined (-1: Half the display refresh rate, 0: Display refresh rate, 1: Custom target FPS, >=2: integer multiplier of the source FPS)
		int iDeltaScalar,
		int iNeighborScalar,
		int iBlackLevel,
		int iWhiteLevel,
		int iSceneChangeThreshold,
		unsigned int iBufferFrames
	) PURE;
};

#ifdef __cplusplus
}
#endif

#endif // __IEZ__
