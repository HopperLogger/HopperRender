#pragma once

#include "iez.h"
#include <chrono>
#include <fstream>
#include <deque>

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
		      public ISpecifyPropertyPages {
  public:
    friend class CCustomInputPin;

    DECLARE_IUNKNOWN;
    static CUnknown* WINAPI CreateInstance(LPUNKNOWN punk, HRESULT* phr);

    // Reveals IHopperRender and ISpecifyPropertyPages
    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv) override;

    // Overridden from CTransformFilter base class
    HRESULT CheckInputType(const CMediaType* mtIn) override;
	HRESULT CheckTransform(const CMediaType* mtIn, const CMediaType* mtOut) override;
    HRESULT DecideBufferSize(IMemAllocator* pAlloc,
			     ALLOCATOR_PROPERTIES* ppropInputRequest) override;
    HRESULT GetMediaType(int iPosition, CMediaType* pMediaType) override;
    HRESULT Transform(IMediaSample* pIn, IMediaSample* pOut) override;
	HRESULT NewSegment(REFERENCE_TIME tStart, REFERENCE_TIME tStop, double dRate) override;
    STDMETHODIMP Stop() override;

    // These implement the custom settings interface
	STDMETHODIMP GetCurrentSettings(bool* pbActivated,
		int* piFrameOutput,
		double* pdTargetFPS,
		bool* pbUseDisplayFPS,
		int* piDeltaScalar,
		int* piNeighborScalar,
		int* piBlackLevel,
		int* piWhiteLevel,
		int* piSceneChangeThreshold,
		int* piIntActiveState,
		double* pdSourceFPS,
		double* pdOFCCalcTime,
	    double* pdAVGOFCCalcTime,
		double* pdPeakOFCCalcTime,
		double* pdWarpCalcTime,
		int* piDimX,
		int* piDimY,
		int* piLowDimX,
		int* piLowDimY,
		unsigned int* piTotalFrameDelta) override;
	STDMETHODIMP UpdateUserSettings(bool bActivated, int iFrameOutput, double dTargetFPS, bool bUseDisplayFPS,
							        int iDeltaScalar, int iNeighborScalar, int iBlackLevel, int iWhiteLevel, int iSceneChangeThreshold) override;

    // ISpecifyPropertyPages interface
    STDMETHODIMP GetPages(CAUUID* pPages) override;

    // CPersistStream override
    STDMETHODIMP GetClassID(CLSID* pClsid) override;

    // Constructor
    CHopperRender(TCHAR* tszName, LPUNKNOWN punk, HRESULT* phr);
    ~CHopperRender();

    HRESULT UpdateVideoInfoHeader(CMediaType* pMediaType);
    HRESULT DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut);
    void autoAdjustSettings();
	HRESULT loadSettings(int* deltaScalar, int* neighborScalar, float* blackLevel, float* whiteLevel, int* maxCalcRes);
    void UpdateInterpolationStatus();
    void useDisplayRefreshRate();
    long CalculateOutputStride();

    // Logging functions
    void InitializeLogging();
    void CloseLogging();
    void LogError(const char* functionName, const char* errorMessage);

    CCritSec m_csHopperRenderLock; // Private play critical section
    CCritSec m_csLogLock; // Critical section for logging

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
	double m_dBlendingScalar; // Blends from frame 1 to frame 2 (0.0 shows 100% frame 1, 1.0 shows 100% frame 2)
	bool m_bUseDisplayFPS; // Whether to use the display refresh rate as target FPS
	bool m_bValidFrameTimes; // Whether valid frame times have been received
	unsigned int m_iSceneChangeThreshold; // Threshold for scene change detection (total frame delta value)
	
	// Peak total frame delta tracking with sliding window
	struct FrameDeltaEntry {
		unsigned int frameNumber;
		unsigned int totalDelta;
	};
	std::deque<FrameDeltaEntry> m_frameDeltaHistory; // History of frame deltas for sliding window
	unsigned int m_iPeakTotalFrameDelta; // Cached peak total frame delta in the last 3 seconds
};
