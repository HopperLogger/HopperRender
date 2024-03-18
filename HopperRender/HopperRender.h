#pragma once

#include <chrono>

#include "opticalFlowCalc.cuh"

#define TEST_MODE 0
#define FT_TARGET 166667 // 60 fps in 100ns units
#define LOG_PERFORMANCE 0
#define MIN_NUM_STEPS 4
#define MAX_NUM_STEPS 1000

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

	// Overriden from CTransformFilter base class
	HRESULT CheckInputType(const CMediaType* mtIn) override;
	HRESULT CheckTransform(const CMediaType* mtIn, const CMediaType* mtOut) override;
	HRESULT DecideBufferSize(IMemAllocator* pAlloc,
	                         ALLOCATOR_PROPERTIES* ppropInputRequest) override;
	HRESULT GetMediaType(int iPosition, CMediaType* pMediaType) override;
	HRESULT CompleteConnect(PIN_DIRECTION dir, IPin* pReceivePin) override;
	HRESULT Transform(IMediaSample* pIn, IMediaSample* pOut) override;
	HRESULT NewSegment(REFERENCE_TIME tStart, REFERENCE_TIME tStop, double dRate) override;

	// These implement the custom SettingsInterface interface
	STDMETHODIMP get_Settings(bool* pbActivated, int* piFrameOutput, int* piNumIterations,
							  int* piFrameBlurKernelSize, int* piFlowBlurKernelSize, int* piIntActiveState, double* pdSourceFPS, int* piNumSteps, int* piDimX,
							  int* piDimY, int* piLowDimX, int* piLowDimY) override;
	STDMETHODIMP put_Settings(bool bActivated, int iFrameOutput, 
							  int iNumIterations, int iFrameBlurKernelSize, int iFlowBlurKernelSize) override;

	// ISpecifyPropertyPages interface
	STDMETHODIMP GetPages(CAUUID* pPages) override;

	// CPersistStream override
	STDMETHODIMP GetClassID(CLSID* pClsid) override;

private:
	// Constructor
	CHopperRender(TCHAR* tszName, LPUNKNOWN punk, HRESULT* phr);

	HRESULT UpdateVideoInfoHeader(CMediaType* pMediaType) const;
	HRESULT DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut, REFERENCE_TIME rtAvgFrameTimeTarget);
	HRESULT CopyFrame(const unsigned char* pInBuffer, unsigned char* pOutBuffer) const;
	HRESULT InterpolateFrame(const unsigned char* pInBuffer, unsigned char* pOutBuffer, float fScalar, int iIntFrameNum);
	void adjustFrameScalar(double dResolutionDivider);
	void autoAdjustSettings();
	HRESULT loadSettings();

	CCritSec m_csHopperRenderLock; // Private play critical section
	bool m_bActivated; // Whether the filter is activated
	bool m_bP010Input; // Whether the input is P010 or not
	int m_iFrameOutput; // What frame output to use
	unsigned char m_cNumTimesTooSlow; // The number of times the interpolation has been too slow
	int m_iNumIterations; // Number of iterations to use in the optical flow calculation
	int m_iNumSteps; // Number of steps executed to find the ideal offset (limits the maximum offset)
	int m_iFrameBlurKernelSize; // The size of the blur kernel for the frames
	int m_iFlowBlurKernelSize; // The size of the blur kernel for the optical flow
	long m_lBufferRequest; // The number of buffers to use
	bool m_bBisNewest; // Which frame order are we using
	OpticalFlowCalc* m_pofcOpticalFlowCalc; // Optical flow calculator
	unsigned int m_iFrameCounter; // Frame counter (relative! i.e. number of frames presented)
	REFERENCE_TIME m_rtCurrStartTime; // The start time of the current interpolated frame
	REFERENCE_TIME m_rtLastStartTime; // The start time of the last interpolated frame
	bool m_bIntNeeded; // Whether interpolation currently is needed or not
	bool m_bIntTooSlow; // Whether the interpolation is too slow
	int m_iNumSamples; // Number of output samples for every input sample
	REFERENCE_TIME m_rtAvgSourceFrameTime; // The average frame time of the source
	REFERENCE_TIME m_rtCurrPlaybackFrameTime; // The current playback frame time
	std::chrono::time_point<std::chrono::high_resolution_clock> m_tpCurrCalcStart; // The start time of the current calculation
	std::chrono::time_point<std::chrono::high_resolution_clock> m_tpCurrCalcEnd; // The end time of the current calculation
	double m_dCurrCalcDuration; // The duration of the current calculation
	bool m_bFirstFrame; // Whether the current frame is the first frame
	double m_dDimScalar; // The scalar to scale the frame dimensions with depending on the renderer used
	unsigned int m_iDimX; // The width of the frame
	unsigned int m_iDimY; // The height of the frame
	double m_dResolutionScalar; // The scalar to scale the resolution with
	double m_dResolutionDivider; // The divider to scale the resolution with
};