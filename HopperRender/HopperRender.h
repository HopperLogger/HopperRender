#pragma once

#include <string>

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

    HRESULT DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut, int iNumSamples) const;
    HRESULT Transform(IMediaSample *pMediaSample);

    CCritSec    m_HopperRenderLock;        // Private play critical section
    int         m_effect;                  // Which effect are we processing
    int m_numSteps;                        // Number of steps executed to find the ideal offset (limits the maximum offset)
    int m_maxOffsetDivider;                // The divider used to calculate the initial global offset
    const long m_lBufferRequest;           // The number of buffers to use
    GPUArray<unsigned char> m_frameA;      // GPU frame A
    GPUArray<unsigned char> m_frameB;      // GPU frame B
    GPUArray<int> m_offsetArray;           // GPU offset array
    GPUArray<unsigned char> m_warpedFrame; // GPU warped frame
    bool m_bAbeforeB;                      // Which frame order are we using
    OpticalFlowCalc m_opticalFlowCalc;     // Optical flow calculator
    int m_frameCounter = 0;                // Frame counter (relative! i.e. number of frames presented)

}; // HopperRender