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
    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void ** ppv);

    // CPersistStream stuff
    HRESULT ScribbleToStream(IStream *pStream);
    HRESULT ReadFromStream(IStream *pStream);

    // Overrriden from CTransformFilter base class

    HRESULT Transform(IMediaSample *pIn, IMediaSample *pOut);
    HRESULT CheckInputType(const CMediaType *mtIn);
    HRESULT CheckTransform(const CMediaType *mtIn, const CMediaType *mtOut);
    HRESULT DecideBufferSize(IMemAllocator *pAlloc,
                             ALLOCATOR_PROPERTIES *pProperties);
    HRESULT GetMediaType(int iPosition, CMediaType *pMediaType);

    // These implement the custom IIPEffect interface

    STDMETHODIMP get_IPEffect(int* IPEffect, int* pNumSteps, int* pMaxOffsetDivider);
    STDMETHODIMP put_IPEffect(int IPEffect, int numSteps, int maxOffsetDivider);

    // ISpecifyPropertyPages interface
    STDMETHODIMP GetPages(CAUUID *pPages);

    // CPersistStream override
    STDMETHODIMP GetClassID(CLSID *pClsid);

private:

    // Constructor
    CHopperRender(TCHAR *tszName, LPUNKNOWN punk, HRESULT *phr);

    // Look after doing the special effect
    BOOL CanPerformHopperRender(const CMediaType *pMediaType) const;
    HRESULT Copy(IMediaSample *pSource, IMediaSample *pDest) const;
    HRESULT Transform(IMediaSample *pMediaSample);

    CCritSec    m_HopperRenderLock;     // Private play critical section
    int         m_effect;               // Which effect are we processing
    int m_numSteps;                     // Number of steps executed to find the ideal offset (limits the maximum offset)
    int m_maxOffsetDivider;             // The divider used to calculate the initial global offset
    const long m_lBufferRequest;        // The number of buffers to use
    GPUArray<unsigned char> m_frameA;   // GPU frame A
    GPUArray<unsigned char> m_frameB;   // GPU frame B
    GPUArray<unsigned char> m_outFrame; // GPU output frame
    bool m_bAbeforeB;                   // Which frame order are we using
    OpticalFlowCalc m_opticalFlowCalc;  // Optical flow calculator
    std::string m_debugMessage = "";    // Debug message
    int m_frameCounter = 0;             // Frame counter (relative! i.e. number of frames presented)

}; // HopperRender