#include "CustomInputPin.h"
#include "CustomAllocator.h"
#include "HopperRender.h"

#include <atlbase.h>

CCustomInputPin::CCustomInputPin(HRESULT* phr, LPCWSTR Name,
					       CHopperRender* pBaseRenderer)
    : CTransformInputPin(NAME("Transform input pin"), pBaseRenderer, phr, Name),
      m_pFilter(pBaseRenderer) {}

STDMETHODIMP CCustomInputPin::NonDelegatingQueryInterface(REFIID riid,
								 void** ppv) {
    CheckPointer(ppv, E_POINTER);
    return __super::NonDelegatingQueryInterface(riid, ppv);
}

STDMETHODIMP CCustomInputPin::GetAllocator(IMemAllocator** ppAllocator) {

    CheckPointer(ppAllocator, E_POINTER);

    if (m_pAllocator) {
	// We already have an allocator, so return that one.
	*ppAllocator = m_pAllocator;
	(*ppAllocator)->AddRef();
	return S_OK;
    }

    // No allocator yet, so propose our custom allocator.
    HRESULT hr = S_OK;
    m_pCustomAllocator = new (std::nothrow)
	CCustomAllocator(L"Custom allocator", nullptr, this, &hr);
    if (!m_pCustomAllocator) {
	return E_OUTOFMEMORY;
    }
    if (FAILED(hr)) {
	delete m_pCustomAllocator;
	return hr;
    }

    if (m_pNewMT) {
	m_pCustomAllocator->SetNewMediaType(*m_pNewMT.get());
	m_pNewMT.reset();
    }

    // Return the IMemAllocator interface to the caller.
    return m_pCustomAllocator->QueryInterface(IID_IMemAllocator,
					      (void**)ppAllocator);
}

STDMETHODIMP
CCustomInputPin::GetAllocatorRequirements(ALLOCATOR_PROPERTIES* pProps) {
    // 1 buffer required
    ZeroMemory(pProps, sizeof(ALLOCATOR_PROPERTIES));
    pProps->cbBuffer = 1;
    return S_OK;
}

STDMETHODIMP
CCustomInputPin::ReceiveConnection(IPin* pConnector,
					  const AM_MEDIA_TYPE* pmt) {
    CAutoLock cObjectLock(m_pLock);

    if (m_Connected) {
		CMediaType mt(*pmt);

		if (FAILED(CheckMediaType(&mt))) {
			return VFW_E_TYPE_NOT_ACCEPTED;
		}

		ALLOCATOR_PROPERTIES props, actual;

		CComPtr<IMemAllocator> pMemAllocator;
		if (FAILED(GetAllocator(&pMemAllocator)) ||
			FAILED(pMemAllocator->Decommit()) ||
			FAILED(pMemAllocator->GetProperties(&props))) {
			return S_OK;
		}

		CMediaType mtNew(*pmt);
		props.cbBuffer = m_pFilter->m_pOutput->CurrentMediaType().GetSampleSize();

		if (FAILED(pMemAllocator->SetProperties(&props, &actual)) ||
			FAILED(pMemAllocator->Commit()) ||
			props.cbBuffer != actual.cbBuffer) {
			return E_FAIL;
		}

		return SetMediaType(&mt) == S_OK // here set mt, not mtNew
			   ? S_OK
			   : VFW_E_TYPE_NOT_ACCEPTED;
	}

    return __super::ReceiveConnection(pConnector, pmt);
}

STDMETHODIMP CCustomInputPin::NewSegment(REFERENCE_TIME startTime,
						REFERENCE_TIME stopTime,
						double rate) {
    CAutoLock cReceiveLock(&m_csReceive);

    m_pFilter->NewSegment(startTime, stopTime, rate);
    return CTransformInputPin::NewSegment(startTime, stopTime, rate);
}

STDMETHODIMP CCustomInputPin::BeginFlush() {
    CAutoLock cReceiveLock(&m_csReceive);
    return CTransformInputPin::BeginFlush();
}

void CCustomInputPin::SetNewMediaType(const CMediaType& mt) {
    m_pNewMT.reset();
    if (m_pCustomAllocator) {
	m_pCustomAllocator->SetNewMediaType(mt);
    } else {
	m_pNewMT.reset(new CMediaType(mt));
    }
}

void CCustomInputPin::ClearNewMediaType() {
    m_pNewMT.reset();
    if (m_pCustomAllocator) {
	m_pCustomAllocator->ClearNewMediaType();
    }
}
