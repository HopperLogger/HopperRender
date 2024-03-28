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
		int* piFrameOutput, // What frame output to use (0: WarpedFrame 1 -> 2, 1: WarpedFrame 2 -> 1, 2: BlendedFrame, 3: HSV Flow, 4: Blurred Frames)
		int* piNumIterations, // Number of iterations to use in the optical flow calculation (0: As many as possible)
		int* piFrameBlurKernelSize, // The size of the blur kernel used to blur the source frames before calculating the optical flow
		int* piFlowBlurKernelSize, // The size of the blur kernel used to blur the offset calculated by the optical flow
		int* piSceneChangeThreshold, // The threshold used to determine whether a scene change has occurred
		int* piCurrentSceneChange, // How many pixel differences are currently detected
		int* piIntActiveState, // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)
		double* pdSourceFPS, // The source frames per second
		int* piNumSteps, // Number of steps executed to find the ideal offset (limits the maximum offset)
		int* piDimX, // The width of the frame
		int* piDimY, // The height of the frame
		int* piLowDimX, // The width of the downscaled frame used in the optical flow calculation
		int* piLowDimY // The height of the downscaled frame used in the optical flow calculation
	) PURE;

	STDMETHOD(UpdateUserSettings)(THIS_
		bool bActivated, // Whether the filter is activated by the user
		int iFrameOutput, // What frame output to use (0: WarpedFrame 1 -> 2, 1: WarpedFrame 2 -> 1, 2: BlendedFrame, 3: HSV Flow, 4: Blurred Frames)
		int iNumIterations, // Number of iterations to use in the optical flow calculation (0: As many as possible)
		int iFrameBlurKernelSize, // The size of the blur kernel used to blur the source frames before calculating the optical flow
		int iFlowBlurKernelSize, // The size of the blur kernel used to blur the offset calculated by the optical flow
		int iSceneChangeThreshold // The threshold used to determine whether a scene change has occurred
	) PURE;
};

#ifdef __cplusplus
}
#endif

#endif // __IEZ__