#pragma once

#include <memory>
#include <mfidl.h>
#include <streams.h>

class CHopperRender;
class CCustomAllocator;

class CCustomInputPin : public CTransformInputPin {
  private:
    friend class CCustomAllocator;

    CHopperRender* m_pFilter;

    CCustomAllocator* m_pCustomAllocator = nullptr;
    std::unique_ptr<CMediaType> m_pNewMT;

    CCritSec m_csReceive;

  public:
    CCustomInputPin(HRESULT* phr, LPCWSTR Name,
			   CHopperRender* pBaseRenderer);
    ~CCustomInputPin() = default;

    // CUnknown
    DECLARE_IUNKNOWN
    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv);

    // CBaseInputPin
    STDMETHODIMP GetAllocator(IMemAllocator** ppAllocator);
    STDMETHODIMP GetAllocatorRequirements(ALLOCATOR_PROPERTIES* pProps);
    STDMETHODIMP ReceiveConnection(IPin* pConnector, const AM_MEDIA_TYPE* pmt);

    STDMETHODIMP NewSegment(REFERENCE_TIME startTime, REFERENCE_TIME stopTime,
			    double rate) override;
    STDMETHODIMP BeginFlush() override;

    void SetNewMediaType(const CMediaType& mt);
    void ClearNewMediaType();

};
