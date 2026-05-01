#include "CustomInputPin.h"
#include "CustomAllocator.h"
#include <dvdmedia.h>
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

		// Determine the buffer size required by the new input format.
		// Take the maximum of all available size hints so the
		// allocator is large enough to actually hold an incoming sample.
		long inputSampleSize = mt.GetSampleSize();
		long inputBiSize = 0;
		long inputBiW = 0, inputBiH = 0;
		if (mt.Format() && mt.FormatType()) {
			if (*mt.FormatType() == FORMAT_VideoInfo) {
				VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)mt.Format();
				inputBiSize = (long)pvi->bmiHeader.biSizeImage;
				inputBiW = pvi->bmiHeader.biWidth;
				inputBiH = pvi->bmiHeader.biHeight;
			} else if (*mt.FormatType() == FORMAT_VideoInfo2) {
				VIDEOINFOHEADER2* pvi2 = (VIDEOINFOHEADER2*)mt.Format();
				inputBiSize = (long)pvi2->bmiHeader.biSizeImage;
				inputBiW = pvi2->bmiHeader.biWidth;
				inputBiH = pvi2->bmiHeader.biHeight;
			}
		}
		const bool isHDR = IsEqualGUID(*mt.Subtype(), MEDIASUBTYPE_P010);
		const long bytesPerSample = isHDR ? 2 : 1;
		const long computedMin = inputBiW * abs(inputBiH) * 3 / 2 * bytesPerSample;
		long requiredInputBuf = max(inputSampleSize, max(inputBiSize, computedMin));
		if (requiredInputBuf <= 0) requiredInputBuf = 1;

		// Commit the new media type to this pin first so the filter sees the
		// new input dimensions before output renegotiation reads them.
		if (SetMediaType(&mt) != S_OK) {
			return VFW_E_TYPE_NOT_ACCEPTED;
		}

		// Update the filter's cached dimensions/stride from the new input MT
		// and tear down any optical flow calculator built for the old size so
		// DeliverToRenderer rebuilds it for the new dimensions.
		const long newDimX = inputBiW;
		const long newDimY = abs(inputBiH);
		if (newDimX > 0 && newDimY > 0) {
			char logMsg[128];
			sprintf_s(logMsg, "Input format change: %ux%u -> %ldx%ld",
				m_pFilter->m_iDimX, m_pFilter->m_iDimY, newDimX, newDimY);
			m_pFilter->Log(LogLevel::Info, "InputPin::ReceiveConnection", logMsg);

			m_pFilter->m_iDimX = static_cast<unsigned int>(newDimX);
			m_pFilter->m_iDimY = static_cast<unsigned int>(newDimY);
			m_pFilter->m_iInputStride = static_cast<unsigned int>(newDimX);

			if (m_pFilter->m_pofcOpticalFlowCalc) {
				delete m_pFilter->m_pofcOpticalFlowCalc;
				m_pFilter->m_pofcOpticalFlowCalc = nullptr;
			}
		}

		// Renegotiate the OUTPUT pin so its allocator (and the downstream
		// renderer) can be sized for the new dimensions before upstream
		// starts pushing larger samples.
		if (m_pFilter->m_pOutput && m_pFilter->m_pOutput->IsConnected()) {
			CMediaType mtOut(m_pFilter->m_pOutput->CurrentMediaType());
			if (SUCCEEDED(m_pFilter->UpdateVideoInfoHeader(&mtOut))) {
				IPin* pDownstream = m_pFilter->m_pOutput->GetConnected();
				if (pDownstream) {
					HRESULT hrOut = pDownstream->ReceiveConnection(m_pFilter->m_pOutput, &mtOut);
					if (SUCCEEDED(hrOut)) {
						m_pFilter->m_pOutput->SetMediaType(&mtOut);

						// Resize the output allocator too.
						IMemInputPin* pMemIn = nullptr;
						if (SUCCEEDED(pDownstream->QueryInterface(IID_IMemInputPin, (void**)&pMemIn)) && pMemIn) {
							IMemAllocator* pOutAlloc = nullptr;
							if (SUCCEEDED(pMemIn->GetAllocator(&pOutAlloc)) && pOutAlloc) {
								ALLOCATOR_PROPERTIES outProps, outActual;
								if (SUCCEEDED(pOutAlloc->Decommit()) &&
									SUCCEEDED(pOutAlloc->GetProperties(&outProps))) {
									outProps.cbBuffer = max(outProps.cbBuffer, (long)mtOut.GetSampleSize());
									if (SUCCEEDED(pOutAlloc->SetProperties(&outProps, &outActual))) {
										pOutAlloc->Commit();
									}
								}
								pOutAlloc->Release();
							}
							pMemIn->Release();
						}
					} else {
						m_pFilter->Log(LogLevel::Error, "InputPin::ReceiveConnection",
							"Downstream rejected output renegotiation");
					}
				}
			}
		}

		// Resize the INPUT allocator for the new sample size.
		ALLOCATOR_PROPERTIES props, actual;
		CComPtr<IMemAllocator> pMemAllocator;
		if (FAILED(GetAllocator(&pMemAllocator)) ||
			FAILED(pMemAllocator->Decommit()) ||
			FAILED(pMemAllocator->GetProperties(&props))) {
			return S_OK;
		}

		props.cbBuffer = requiredInputBuf;
		if (FAILED(pMemAllocator->SetProperties(&props, &actual)) ||
			FAILED(pMemAllocator->Commit()) ||
			actual.cbBuffer < requiredInputBuf) {
			m_pFilter->Log(LogLevel::Error, "InputPin::ReceiveConnection",
				"Input allocator could not satisfy required buffer size");
			return E_FAIL;
		}

		return S_OK;
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
