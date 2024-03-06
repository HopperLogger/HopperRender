#ifndef __IEZ__
#define __IEZ__

#ifdef __cplusplus
extern "C" {
#endif

// { fd5010a3-8ebe-11ce-8183-00aa00577da1 }
DEFINE_GUID(IID_SettingsInterface,
            0xfd5010a3, 0x8ebe, 0x11ce, 0x81, 0x83, 0x00, 0xaa, 0x00, 0x57, 0x7d, 0xa1);

DECLARE_INTERFACE_(SettingsInterface, IUnknown) {
	STDMETHOD(get_Settings)(THIS_
		bool* pbActivated, // Is the filter activated
		int* piFrameOutput, // What frame output to use
		int* piNumIterations, // Number of iterations to find the ideal offset
		int* piBlurKernelSize, // The size of the blur kernel
		int* piIntActiveState, // Is the effect active
		double* pdSourceFPS, // The source frames per second
		int* piNumSteps, // Number of steps executed to find the ideal offset (limits the maximum offset)
		int* piDimX, // The width of the frame
		int* piDimY, // The height of the frame
		int* piLowDimX, // The width of the low resolution frame
		int* piLowDimY // The height of the low resolution frame
	) PURE;

	STDMETHOD(put_Settings)(THIS_
		bool bActivated, // Is the filter activated
		int iFrameOutput, // What frame output to use
		int iNumSteps, // Number of steps executed to find the ideal offset (limits the maximum offset)
		int iBlurKernelSize // The size of the blur kernel
	) PURE;
};

#ifdef __cplusplus
}
#endif

#endif // __IEZ__