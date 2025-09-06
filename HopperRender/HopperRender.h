#pragma once

#include <chrono>
#include "iez.h"

#include "opticalFlowCalc.h"

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
    friend class CCustomInputPin;

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
		double* pdTargetFPS,
		int* piDeltaScalar,
		int* piNeighborScalar,
		int* piBlackLevel,
		int* piWhiteLevel,
		int* piIntActiveState,
		double* pdSourceFPS,
		double* pdOFCCalcTime,
		double* pdWarpCalcTime,
		int* piDimX,
		int* piDimY,
		int* piLowDimX,
		int* piLowDimY) override;
	STDMETHODIMP UpdateUserSettings(bool bActivated, int iFrameOutput, double dTargetFPS,
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
	HRESULT loadSettings(int* deltaScalar, int* neighborScalar, float* blackLevel, float* whiteLevel, int* maxCalcRes);
	void UpdateInterpolationStatus();
	void useDisplayRefreshRate();

	CCritSec m_csHopperRenderLock; // Private play critical section

	// Settings
	FrameOutput m_iFrameOutput; // What frame output to use (0: WarpedFrame 1 -> 2, 1: WarpedFrame 2 -> 1, 2: BlendedFrame, 3: HSV Flow, 4: Blurred Frames, 5: Side-by-side 1, 6: Side-by-side 2)
	
	// Video info
	unsigned int m_iDimX; // The width of the frame
	unsigned int m_iDimY; // The height of the frame

	// Timings
	REFERENCE_TIME m_rtCurrStartTime; // The start time of the current interpolated frame
	REFERENCE_TIME m_rtSourceFrameTime; // The average frame time of the source frames (at 1.0x speed)
	REFERENCE_TIME m_rtTargetFrameTime; // The frame time that will be interpolated to
	REFERENCE_TIME m_rtCurrPlaybackFrameTime; // The current frame time of the source frames (accounting for the current playback speed)

	// Optical flow calculation
	OpticalFlowCalc* m_pofcOpticalFlowCalc; // Optical flow calculator

	// Frame output
	unsigned int m_iFrameCounter; // Frame counter (relative! i.e. number of source frames received so far)
	int m_iNumIntFrames; // Number of output samples (interpolated frames) for every input sample (source frame)

	// Performance and activation status
	ActiveState m_iIntActiveState; // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)

    double m_dTotalWarpDuration; // The total duration of the current frame warp
	double blendingScalar; // Blends from frame 1 to frame 2 (0.0 shows 100% frame 1, 1.0 shows 100% frame 2)
};
