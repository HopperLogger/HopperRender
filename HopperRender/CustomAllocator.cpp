#include "CustomAllocator.h"
#include "CustomInputPin.h"
#include "dvdmedia.h"

CCustomMediaSample::CCustomMediaSample(LPCTSTR pName,
				       CBaseAllocator* pAllocator, HRESULT* phr,
				       LPBYTE pBuffer, LONG length)
    : CMediaSampleSideData(pName, pAllocator, phr, pBuffer, length) {}

CCustomAllocator::CCustomAllocator(
    LPCTSTR pName, LPUNKNOWN pUnk,
    CCustomInputPin* pCustomInputPin, HRESULT* phr)
    : CMemAllocator(pName, nullptr, phr),
      m_pCustomInputPin(pCustomInputPin) {}

CCustomAllocator::~CCustomAllocator() {
    if (m_pCustomInputPin &&
	m_pCustomInputPin->m_pCustomAllocator == this) {
	m_pCustomInputPin->m_pCustomAllocator = nullptr;
    }
}

HRESULT CCustomAllocator::Alloc(void) {
    CAutoLock lck(this);

    /* Check he has called SetProperties */
    HRESULT hr = CBaseAllocator::Alloc();
    if (FAILED(hr)) {
	return hr;
    }

    /* If the requirements haven't changed then don't reallocate */
    if (hr == S_FALSE) {
	ASSERT(m_pBuffer);
	return NOERROR;
    }
    ASSERT(hr == S_OK); // we use this fact in the loop below

    /* Free the old resources */
    if (m_pBuffer) {
	ReallyFree();
    }

    /* Make sure we've got reasonable values */
    if (m_lSize < 0 || m_lPrefix < 0 || m_lCount < 0) {
	return E_OUTOFMEMORY;
    }

    /* Compute the aligned size */
    LONG lAlignedSize = m_lSize + m_lPrefix;

    /*  Check overflow */
    if (lAlignedSize < m_lSize) {
	return E_OUTOFMEMORY;
    }

    if (m_lAlignment > 1) {
	LONG lRemainder = lAlignedSize % m_lAlignment;
	if (lRemainder != 0) {
	    LONG lNewSize = lAlignedSize + m_lAlignment - lRemainder;
	    if (lNewSize < lAlignedSize) {
		return E_OUTOFMEMORY;
	    }
	    lAlignedSize = lNewSize;
	}
    }

    /* Create the contiguous memory block for the samples
       making sure it's properly aligned (64K should be enough!)
    */
    ASSERT(lAlignedSize % m_lAlignment == 0);

    LONGLONG lToAllocate = m_lCount * (LONGLONG)lAlignedSize;

    /*  Check overflow */
    if (lToAllocate > MAXLONG) {
	return E_OUTOFMEMORY;
    }

    m_pBuffer = (PBYTE)VirtualAlloc(NULL, (LONG)lToAllocate, MEM_COMMIT,
				    PAGE_READWRITE);

    if (m_pBuffer == NULL) {
	return E_OUTOFMEMORY;
    }

    LPBYTE pNext = m_pBuffer;
    CCustomMediaSample* pSample;

    ASSERT(m_lAllocated == 0);

    // Create the new samples - we have allocated m_lSize bytes for each sample
    // plus m_lPrefix bytes per sample as a prefix. We set the pointer to
    // the memory after the prefix - so that GetPointer() will return a pointer
    // to m_lSize bytes.
    for (; m_lAllocated < m_lCount; m_lAllocated++, pNext += lAlignedSize) {
	pSample = new (std::nothrow)
	    CCustomMediaSample(L"Custom media sample", this, &hr,
			       pNext + m_lPrefix, // GetPointer() value
			       m_lSize);          // not including prefix

	ASSERT(SUCCEEDED(hr));
	if (pSample == NULL) {
	    return E_OUTOFMEMORY;
	}

	// This CANNOT fail
	m_lFree.Add(pSample);
    }

    m_bChanged = FALSE;
    return NOERROR;
}

STDMETHODIMP
CCustomAllocator::SetProperties(__in ALLOCATOR_PROPERTIES* pRequest,
				__out ALLOCATOR_PROPERTIES* pActual) {
    CheckPointer(pActual, E_POINTER);
    ValidateReadWritePtr(pActual, sizeof(ALLOCATOR_PROPERTIES));

    ASSERT(pRequest->cbBuffer > 0);

    if (m_cbBuffer) {
	pRequest->cbBuffer = m_cbBuffer;
	m_cbBuffer = 0;
    }

    return __super::SetProperties(pRequest, pActual);
}

HRESULT CCustomAllocator::GetBuffer(IMediaSample** ppBuffer,
				    REFERENCE_TIME* pStartTime,
				    REFERENCE_TIME* pEndTime, DWORD dwFlags) {
    HRESULT hr = __super::GetBuffer(ppBuffer, pStartTime, pEndTime, dwFlags);

    if (SUCCEEDED(hr) && m_pNewMT) {
	(*ppBuffer)->SetMediaType(m_pNewMT.get());
    }

    return hr;
}

BITMAPINFOHEADER* GetBIHfromVIHs(const AM_MEDIA_TYPE* pmt) {
    if (pmt->formattype == FORMAT_VideoInfo2) {
	return &((VIDEOINFOHEADER2*)pmt->pbFormat)->bmiHeader;
    }

    if (pmt->formattype == FORMAT_VideoInfo) {
	return &((VIDEOINFOHEADER*)pmt->pbFormat)->bmiHeader;
    }

    return nullptr;
}

void CCustomAllocator::SetNewMediaType(const CMediaType& mt) {
    m_pNewMT.reset(new CMediaType(mt));

    m_cbBuffer = 0;
    if (const auto pBIH = GetBIHfromVIHs(m_pNewMT.get()); pBIH) {
	m_cbBuffer = pBIH->biSizeImage ? pBIH->biSizeImage : DIBSIZE(*pBIH);
    }
}

void CCustomAllocator::ClearNewMediaType() {
    m_pNewMT.reset();
    m_cbBuffer = 0;
}
