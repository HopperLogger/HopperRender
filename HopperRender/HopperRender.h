#pragma once

#include "GPUArrayLib.cuh"
#include "opticalFlowCalc.cuh"

class CHopperRender : public CTransformFilter,
         public IIPEffect,
         public ISpecifyPropertyPages,
         public CPersistStream
{

public:

    DECLARE_IUNKNOWN;
    static CUnknown * WINAPI CreateInstance(LPUNKNOWN punk, HRESULT *phr);

    // Reveals IHopperRender and ISpecifyPropertyPages
    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void ** ppv) override;

    // CPersistStream stuff
    HRESULT ScribbleToStream(IStream *pStream) const;
    HRESULT ReadFromStream(IStream *pStream) override;

    // Overriden from CTransformFilter base class
    HRESULT CheckInputType(const CMediaType *mtIn) override;
    HRESULT CheckTransform(const CMediaType *mtIn, const CMediaType *mtOut) override;
    HRESULT DecideBufferSize(IMemAllocator *pAlloc,
                             ALLOCATOR_PROPERTIES *ppropInputRequest) override;
    HRESULT GetMediaType(int iPosition, CMediaType *pMediaType) override;
	HRESULT Transform(IMediaSample *pIn, IMediaSample *pOut) override;

    // These implement the custom IIPEffect interface
    STDMETHODIMP get_IPEffect(int* IPEffect, int* pNumSteps, int* pMaxOffsetDivider) override;
    STDMETHODIMP put_IPEffect(int IPEffect, int numSteps, int maxOffsetDivider) override;

    // ISpecifyPropertyPages interface
    STDMETHODIMP GetPages(CAUUID *pPages) override;

    // CPersistStream override
    STDMETHODIMP GetClassID(CLSID *pClsid) override;

private:

    // Constructor
    CHopperRender(TCHAR *tszName, LPUNKNOWN punk, HRESULT *phr);

    HRESULT DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut, REFERENCE_TIME rtAvgFrameTimeTarget);
    HRESULT InterpolateFrame(BYTE* pInBuffer, BYTE* pOutBuffer, int iIntFrameNum, int iNumSamples);

    CCritSec    m_csHopperRenderLock;                  // Private play critical section
    int         m_iEffect;                           // Which effect are we processing
    int m_iNumSteps;                                 // Number of steps executed to find the ideal offset (limits the maximum offset)
    int m_iMaxOffsetDivider;                         // The divider used to calculate the initial global offset
    const long m_lBufferRequest;                     // The number of buffers to use
    GPUArray<unsigned char> m_gpuFrameA;             // GPU frame A
    GPUArray<unsigned char> m_gpuFrameB;             // GPU frame B
    bool m_bBisNewest;                               // Which frame order are we using
    OpticalFlowCalc m_ofcOpticalFlowCalc;          // Optical flow calculator
    int m_iFrameCounter = 0;                         // Frame counter (relative! i.e. number of frames presented)
    REFERENCE_TIME m_rtCurrStartTime = LONGLONG_MAX; // The start time of the current interpolated frame
	REFERENCE_TIME m_rtLastStartTime = LONGLONG_MAX; // The start time of the last input frame
    BOOL m_bIntNeeded = false;                       // Whether interpolation currently is needed or not
    REFERENCE_TIME m_rtPastFrameDurations[10];       // The past 10 frame durations
    REFERENCE_TIME m_rtAvgSourceFrameTime = 0;       // The average frame time of the source
};