#pragma once

#include <chrono>

#include "opticalFlowCalc.cuh"

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
	HRESULT Transform(IMediaSample* pIn, IMediaSample* pOut) override;
	HRESULT NewSegment(REFERENCE_TIME tStart, REFERENCE_TIME tStop, double dRate) override;

	// These implement the custom SettingsInterface interface
	STDMETHODIMP get_Settings(bool* pbActivated, int* piFrameOutput, int* piNumIterations, int* piBlurKernelSize,
	                          int* piIntActiveState, double* pdSourceFPS, int* piNumSteps) override;
	STDMETHODIMP put_Settings(bool bActivated, int iFrameOutput, int iNumIterations, int iBlurKernelSize) override;

	// ISpecifyPropertyPages interface
	STDMETHODIMP GetPages(CAUUID* pPages) override;

	// CPersistStream override
	STDMETHODIMP GetClassID(CLSID* pClsid) override;

private:
	// Constructor
	CHopperRender(TCHAR* tszName, LPUNKNOWN punk, HRESULT* phr);

	HRESULT DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut, REFERENCE_TIME rtAvgFrameTimeTarget);
	HRESULT InterpolateFrame(BYTE* pInBuffer, BYTE* pOutBuffer, double dScalar, int iIntFrameNum);

	CCritSec m_csHopperRenderLock; // Private play critical section
	bool m_bActivated; // Whether the filter is activated
	int m_iFrameOutput; // What frame output to use
	int m_iNumIterations; // Number of iterations to use in the optical flow calculation
	int m_iNumSteps; // Number of steps executed to find the ideal offset (limits the maximum offset)
	int m_iBlurKernelSize; // The size of the blur kernel
	const long m_lBufferRequest; // The number of buffers to use
	bool m_bBisNewest; // Which frame order are we using
	OpticalFlowCalc m_ofcOpticalFlowCalc; // Optical flow calculator
	int m_iFrameCounter; // Frame counter (relative! i.e. number of frames presented)
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
};