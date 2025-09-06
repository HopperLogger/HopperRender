#pragma once

#include "MediaSampleSideData.h"
#include <memory>

class CCustomInputPin;

class CCustomMediaSample : public CMediaSampleSideData {
  public:
    CCustomMediaSample(LPCTSTR pName, CBaseAllocator* pAllocator, HRESULT* phr,
		       LPBYTE pBuffer, LONG length);
    ~CCustomMediaSample() = default;

    STDMETHODIMP_(ULONG) AddRef() { return __super::AddRef(); }
};

class CCustomAllocator : public CMemAllocator {
  protected:
    HRESULT Alloc();

    CCustomInputPin* m_pCustomInputPin = nullptr;
    std::unique_ptr<CMediaType> m_pNewMT;
    long m_cbBuffer = 0;

  public:
    CCustomAllocator(LPCTSTR pName, LPUNKNOWN pUnk,
		     CCustomInputPin* pCustomInputPin,
		     HRESULT* phr);
    ~CCustomAllocator();

    STDMETHODIMP SetProperties(__in ALLOCATOR_PROPERTIES* pRequest,
			       __out ALLOCATOR_PROPERTIES* pActual);
    STDMETHODIMP GetBuffer(IMediaSample** ppBuffer, REFERENCE_TIME* pStartTime,
			   REFERENCE_TIME* pEndTime, DWORD dwFlags);

    void SetNewMediaType(const CMediaType& mt);
    void ClearNewMediaType();
};
