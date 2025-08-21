#pragma once

#include <chrono>

#include "opticalFlowCalc.h"

#define TEST_MODE 0 // Enables or disables automatic settings change to allow accurate performance testing (0: Disabled, 1: Enabled)
#define FT_TARGET 166667 // The target frame rate in 100ns units (e.g., 166667 for 60 fps)
#define LOG_PERFORMANCE 0 // Whether or not to print debug messages regarding calculation performance (0: Disabled, 1: Enabled)

typedef enum FrameOutput {
    WarpedFrame12,
    WarpedFrame21,
	BlendedFrame,
	HSVFlow,
	GreyFlow,
	SideBySide1,
	SideBySide2
} FrameOutput;

typedef enum ActiveState {
    Deactivated,
	NotNeeded,
	Active,
	TooSlow
} ActiveState;

class CHopperRender : public CTransformFilter,
                      public SettingsInterface,
                      public ISpecifyPropertyPages,
                      public CPersistStream {
public:
	DECLARE_IUNKNOWN;
	static CUnknown* WINAPI CreateInstance(LPUNKNOWN punk, HRESULT* phr);

	// Reveals IHopperRender and ISpecifyPropertyPages
	STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv) override;

	// CPersistStream stuff
	HRESULT ScribbleToStream(IStream* pStream) const;
	HRESULT ReadFromStream(IStream* pStream) override;

	// Overridden from CTransformFilter base class
	HRESULT CheckInputType(const CMediaType* mtIn) override;
	HRESULT CheckTransform(const CMediaType* mtIn, const CMediaType* mtOut) override;
	HRESULT DecideBufferSize(IMemAllocator* pAlloc,
	                         ALLOCATOR_PROPERTIES* ppropInputRequest) override;
	HRESULT GetMediaType(int iPosition, CMediaType* pMediaType) override;
	HRESULT CompleteConnect(PIN_DIRECTION dir, IPin* pReceivePin) override;
	HRESULT Transform(IMediaSample* pIn, IMediaSample* pOut) override;
	HRESULT NewSegment(REFERENCE_TIME tStart, REFERENCE_TIME tStop, double dRate) override;

	// These implement the custom settings interface
	STDMETHODIMP GetCurrentSettings(bool* pbActivated,
		int* piFrameOutput,
		int* piDeltaScalar,
		int* piNeighborScalar,
		int* piBlackLevel,
		int* piWhiteLevel,
		int* piIntActiveState,
		double* pdSourceFPS,
		double* pdTargetFPS,
		double* pdOFCCalcTime,
		double* pdWarpCalcTime,
		int* piDimX,
		int* piDimY,
		int* piLowDimX,
		int* piLowDimY) override;
	STDMETHODIMP UpdateUserSettings(bool bActivated, int iFrameOutput, 
							        int iDeltaScalar, int iNeighborScalar, int iBlackLevel, int iWhiteLevel) override;

	// ISpecifyPropertyPages interface
	STDMETHODIMP GetPages(CAUUID* pPages) override;

	// CPersistStream override
	STDMETHODIMP GetClassID(CLSID* pClsid) override;

	// Constructor
	CHopperRender(TCHAR* tszName, LPUNKNOWN punk, HRESULT* phr);

	HRESULT UpdateVideoInfoHeader(CMediaType* pMediaType);
	HRESULT DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut);
	void autoAdjustSettings();
	HRESULT loadSettings();
	void UpdateInterpolationStatus();

	CCritSec m_csHopperRenderLock; // Private play critical section

	// Settings
	FrameOutput m_iFrameOutput; // What frame output to use (0: WarpedFrame 1 -> 2, 1: WarpedFrame 2 -> 1, 2: BlendedFrame, 3: HSV Flow, 4: Blurred Frames, 5: Side-by-side 1, 6: Side-by-side 2)
	int m_iNumIterations; // Number of iterations to use in the optical flow calculation (0: As many as possible)
	int m_iFrameBlurKernelSize; // The size of the blur kernel used to blur the source frames before calculating the optical flow
	int m_iFlowBlurKernelSize; // The size of the blur kernel used to blur the offset calculated by the optical flow
	
	// Video info
	unsigned int m_iDimX; // The width of the frame
	unsigned int m_iDimY; // The height of the frame
	bool m_bInterlaced; // Whether the video is interlaced or not
	bool m_bHDR; // Whether the video has HDR (BT.2020) or not

	// Timings
	REFERENCE_TIME m_rtCurrStartTime; // The start time of the current interpolated frame
	REFERENCE_TIME m_rtSourceFrameTime; // The average frame time of the source frames (at 1.0x speed)
	REFERENCE_TIME m_rtTargetFrameTime;
	REFERENCE_TIME m_rtCurrPlaybackFrameTime; // The current frame time of the source frames (accounting for the current playback speed)
	std::chrono::time_point<std::chrono::high_resolution_clock> m_tpCurrCalcStart; // The start time of the current calculation
	std::chrono::time_point<std::chrono::high_resolution_clock> m_tpCurrCalcEnd; // The end time of the current calculation
	double m_dCurrCalcDuration; // The duration of the current calculation

	// Optical flow calculation
	OpticalFlowCalc* m_pofcOpticalFlowCalc; // Optical flow calculator
	unsigned char m_cResolutionStep; // Determines which predefined resolution scalar will be used for the optical flow calculation
	float m_fResolutionDividers[6]; // The predefined resolution dividers
	float m_fResolutionScalar; // The scalar to scale the resolution with (only used in the optical flow calculation)
	float m_fResolutionDivider; // The divider to scale the resolution with (only used in the optical flow calculation)

	// Frame output
	double m_dDimScalar; // The scalar to scale the frame dimensions with depending on the renderer used
	unsigned int m_iFrameCounter; // Frame counter (relative! i.e. number of source frames received so far)
	int m_iNumIntFrames; // Number of output samples (interpolated frames) for every input sample (source frame)
	long m_lBufferRequest; // The number of buffers to use in the output buffer queue

	// Performance and activation status
	unsigned char m_cNumTimesTooSlow; // The number of times the interpolation has been too slow
	ActiveState m_iIntActiveState; // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)
	bool m_bExportMode; // Whether the filter is in export mode or not
	bool m_bFirstErrorMessage; // Whether the first error message has been displayed or not

    double totalWarpDuration; // The total duration of the current frame warp
	double blendingScalar;
};
