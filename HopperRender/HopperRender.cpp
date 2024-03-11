#include <windows.h>
#include <streams.h>
#include <string>
#include <initguid.h>

#if (1100 > _MSC_VER)
#include <olectlid.h>
#else
#include <olectl.h>
#endif

#include "uids.h"
#include "iez.h"
#include "HopperRenderSettings.h"
#include "HopperRender.h"

#include <cuda_runtime_api.h>
#include <dvdmedia.h>
#include <numeric>

#include "opticalFlowCalcSDR.cuh"
#include "opticalFlowCalcHDR.cuh"
#include "resource.h"


// Debug message function
void DebugMessage(const std::string& message, const bool showLog) {
	if (showLog) {
		const std::string m_debugMessage = message + "\n";
		OutputDebugStringA(m_debugMessage.c_str());
	}
}


// Input pin types
constexpr AMOVIESETUP_MEDIATYPE sudPinTypesIn =
{
	&MEDIATYPE_Video, // Major type
	&MEDIASUBTYPE_NULL // Minor type
};


// Output pin types
constexpr AMOVIESETUP_MEDIATYPE sudPinTypesOut =
{
	&MEDIATYPE_Video, // Major type
	&MEDIASUBTYPE_P010 // Minor type
};


// Input/Output pin information
constexpr AMOVIESETUP_PIN sudpPins[] =
{
	{
		L"Input", // Pins string name
		FALSE, // Is it rendered
		FALSE, // Is it an output
		FALSE, // Are we allowed none
		FALSE, // And allowed many
		&CLSID_NULL, // Connects to filter
		nullptr, // Connects to pin
		1, // Number of types
		&sudPinTypesIn // Pin information
	},
	{
		L"Output", // Pins string name
		FALSE, // Is it rendered
		TRUE, // Is it an output
		FALSE, // Are we allowed none
		FALSE, // And allowed many
		&CLSID_NULL, // Connects to filter
		nullptr, // Connects to pin
		1, // Number of types
		&sudPinTypesOut // Pin information
	}
};


// Filter information
constexpr AMOVIESETUP_FILTER sudHopperRender =
{
	&CLSID_HopperRender, // Filter CLSID
	L"HopperRender", // String name
	MERIT_DO_NOT_USE, // Filter merit
	2, // Number of pins
	sudpPins // Pin information
};


// List of class IDs and creator functions for the class factory. This
// provides the link between the OLE entry point in the DLL and an object
// being created. The class factory will call the static CreateInstance
CFactoryTemplate g_Templates[] = {
	{L"HopperRender", &CLSID_HopperRender, CHopperRender::CreateInstance, nullptr, &sudHopperRender},
	{L"HopperRender Settings", &CLSID_HopperRenderSettings, CHopperRenderSettings::CreateInstance}
};
int g_cTemplates = sizeof(g_Templates) / sizeof(g_Templates[0]);


// Handles sample registry
STDAPI DllRegisterServer() {
	return AMovieDllRegisterServer2(TRUE);
}


// Handles sample unregistry
STDAPI DllUnregisterServer() {
	return AMovieDllRegisterServer2(FALSE);
}


// DllEntryPoint
extern "C" BOOL WINAPI DllEntryPoint(HINSTANCE, ULONG, LPVOID);

BOOL APIENTRY DllMain(HANDLE hModule,
                      DWORD dwReason,
                      LPVOID lpReserved) {
	return DllEntryPoint(static_cast<HINSTANCE>(hModule), dwReason, lpReserved);
}


// Constructor
CHopperRender::CHopperRender(TCHAR* tszName,
                             LPUNKNOWN punk,
                             HRESULT* phr) :
	CTransformFilter(tszName, punk, CLSID_HopperRender),
	CPersistStream(punk, phr),
	m_bActivated(true),
	m_bP010Input(false),
	m_iFrameOutput(2),
	m_cNumTimesTooSlow(0),
	m_iNumIterations(0),
	m_iNumSteps(10),
	m_iFrameBlurKernelSize(20),
	m_iFlowBlurKernelSize(14),
	m_lBufferRequest(1),
	m_bBisNewest(true),
	m_iFrameCounter(0),
	m_rtCurrStartTime(-1),
	m_rtLastStartTime(-1),
	m_bIntNeeded(false),
	m_bIntTooSlow(false),
	m_iNumSamples(1),
	m_rtAvgSourceFrameTime(0),
	m_rtCurrPlaybackFrameTime(0),
	m_dCurrCalcDuration(0),
	m_bFirstFrame(true),
	m_dDimScalar(1.0),
	m_iDimX(1),
	m_iDimY(1),
	m_dResolutionScalar(4.0),
	m_dResolutionDivider(0.25) {
	loadSettings();
}


// Provide the way for COM to create a HopperRender object
CUnknown* CHopperRender::CreateInstance(LPUNKNOWN punk, HRESULT* phr) {
	ASSERT(phr);

	auto pNewObject = new CHopperRender(NAME("HopperRender"), punk, phr);

	if (pNewObject == nullptr) {
		if (phr)
			*phr = E_OUTOFMEMORY;
	}
	return pNewObject;
}


// Reveals IIPEffect and ISpecifyPropertyPages
STDMETHODIMP CHopperRender::NonDelegatingQueryInterface(REFIID riid, void** ppv) {
	CheckPointer(ppv, E_POINTER)

	if (riid == IID_SettingsInterface) {
		return GetInterface(static_cast<SettingsInterface*>(this), ppv);
	}
	if (riid == IID_ISpecifyPropertyPages) {
		return GetInterface(static_cast<ISpecifyPropertyPages*>(this), ppv);
	}
	return CTransformFilter::NonDelegatingQueryInterface(riid, ppv);
}


// Checks whether a specified media type is acceptable for input
HRESULT CHopperRender::CheckInputType(const CMediaType* mtIn) {
	CheckPointer(mtIn, E_POINTER)

	// Check this is a VIDEOINFOHEADER2 type
	if (*mtIn->FormatType() != FORMAT_VideoInfo2) {
		return E_INVALIDARG;
	}

	// Can we transform this type
	if (IsEqualGUID(*mtIn->Type(), MEDIATYPE_Video) && (IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_NV12) || IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_P010))) {
		return NOERROR;
	}
	return E_FAIL;
}


// Checks whether an input media type is compatible with an output media type
HRESULT CHopperRender::CheckTransform(const CMediaType* mtIn, const CMediaType* mtOut) {
	CheckPointer(mtIn, E_POINTER)
	CheckPointer(mtOut, E_POINTER)

	if (IsEqualGUID(*mtIn->Type(), MEDIATYPE_Video) && (IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_NV12) || IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_P010)) && IsEqualGUID(*mtOut->Subtype(), MEDIASUBTYPE_P010)) {
		return NOERROR;
	}
	return E_FAIL;
}


// Sets the output pin's buffer requirements
HRESULT CHopperRender::DecideBufferSize(IMemAllocator* pAlloc, ALLOCATOR_PROPERTIES* ppropInputRequest) {
	// Is the input pin connected
	if (m_pInput->IsConnected() == FALSE) {
		return E_UNEXPECTED;
	}

	CheckPointer(pAlloc, E_POINTER)
	CheckPointer(ppropInputRequest, E_POINTER)
	HRESULT hr = NOERROR;

	ppropInputRequest->cBuffers = 4;
	ppropInputRequest->cbBuffer = m_pOutput->CurrentMediaType().GetSampleSize() * m_lBufferRequest;
	ppropInputRequest->cbAlign = 1;
	ppropInputRequest->cbPrefix = 0;
	ASSERT(ppropInputRequest->cbBuffer);

	// Ask the allocator to reserve us some sample memory, NOTE the function
	// can succeed (that is return NOERROR) but still not have allocated the
	// memory that we requested, so we must check we got whatever we wanted
	ALLOCATOR_PROPERTIES Actual;
	hr = pAlloc->SetProperties(ppropInputRequest, &Actual);
	if (FAILED(hr)) {
		return hr;
	}

	ASSERT(Actual.cBuffers == 4);

	if (ppropInputRequest->cBuffers > Actual.cBuffers ||
		ppropInputRequest->cbBuffer > Actual.cbBuffer) {
		return E_FAIL;
	}
	return NOERROR;
}


// Retrieves a preferred media type for the output pin
HRESULT CHopperRender::GetMediaType(int iPosition, CMediaType* pMediaType) {
	if (iPosition < 0) {
		return E_INVALIDARG;
	}

	if (iPosition > 0) {
		return VFW_S_NO_MORE_ITEMS;
	}

	CheckPointer(pMediaType, E_POINTER);

	pMediaType->SetType(&MEDIATYPE_Video);
	pMediaType->SetFormatType(&FORMAT_VideoInfo2);
	pMediaType->SetSubtype(&MEDIASUBTYPE_P010);

	UpdateVideoInfoHeader(pMediaType);

	return NOERROR;
}


// Updates the video info header with the information for the output pin
HRESULT CHopperRender::UpdateVideoInfoHeader(CMediaType* pMediaType) const {
	// Get the input media type information
	CMediaType* mtIn = &m_pInput->CurrentMediaType();
	auto pvi = (VIDEOINFOHEADER2*)mtIn->Format();
	const long biWidth = pvi->bmiHeader.biWidth;
	const long biHeight = pvi->bmiHeader.biHeight;
	const unsigned int dwX = pvi->dwPictAspectRatioX;
	const unsigned int dwY = pvi->dwPictAspectRatioY;
	const bool bInterlaced = (pvi->dwInterlaceFlags & AMINTERLACE_IsInterlaced) != 0;
	const GUID guid = MEDIASUBTYPE_P010;

	// Set the VideoInfoHeader2 information
	VIDEOINFOHEADER2* vih2 = (VIDEOINFOHEADER2*)pMediaType->ReallocFormatBuffer(sizeof(VIDEOINFOHEADER2));
	memset(vih2, 0, sizeof(VIDEOINFOHEADER2));
	vih2->rcSource.right = vih2->rcTarget.right = biWidth;
	vih2->rcSource.bottom = vih2->rcTarget.bottom = biHeight;
	vih2->AvgTimePerFrame = pvi->AvgTimePerFrame;
	vih2->dwPictAspectRatioX = dwX;
	vih2->dwPictAspectRatioY = dwY;
	vih2->dwBitRate = pvi->dwBitRate;
	vih2->dwBitErrorRate = pvi->dwBitErrorRate;
	vih2->dwCopyProtectFlags = pvi->dwCopyProtectFlags;
	if (bInterlaced) {
		vih2->dwInterlaceFlags = AMINTERLACE_IsInterlaced | AMINTERLACE_DisplayModeBobOrWeave;
	} else {
		vih2->dwInterlaceFlags = AMINTERLACE_1FieldPerSample;
	}

	// Set the BitmapInfoHeader information
	BITMAPINFOHEADER* pBIH = nullptr;
	pBIH = &vih2->bmiHeader;
	pBIH->biSize = sizeof(BITMAPINFOHEADER);
	pBIH->biWidth = biWidth;
	pBIH->biHeight = biHeight;
	pBIH->biBitCount = 24;
	pBIH->biPlanes = 1;
	pBIH->biSizeImage = (biWidth * biHeight * 12) >> 3;
	pBIH->biCompression = guid.Data1;

	// Set the media type information
	pMediaType->SetSampleSize((biWidth * biHeight * 24) >> 3);
	pMediaType->SetTemporalCompression(0);

	return NOERROR;
}


// Retrieves the filter's settings
IBaseFilter* GetFilterFromPin(IPin* pPin)
{
	CheckPointer(pPin, nullptr);

	PIN_INFO pi;
	if (pPin && SUCCEEDED(pPin->QueryPinInfo(&pi)))
	{
		return pi.pFilter;
	}

	return nullptr;
}


// Completes the pin connection process
HRESULT CHopperRender::CompleteConnect(PIN_DIRECTION dir, IPin* pReceivePin) {
	if (dir == PINDIR_OUTPUT) {
		// Check if we're connecting to madVR
		IBaseFilter* pFilter = GetFilterFromPin(pReceivePin);
		if (pFilter != nullptr) {
			CLSID guid;
			if (SUCCEEDED(pFilter->GetClassID(&guid))) {
				if (guid == CLSID_madVR) {
					m_dDimScalar = 16.0 / 15.0;
				} else {
					m_dDimScalar = 1.0;
				}
			}
			pFilter->Release();
		}

		// Check if the input is P010
		m_pInput->CurrentMediaType().subtype == MEDIASUBTYPE_P010 ? m_bP010Input = true : m_bP010Input = false;

		// Get the frame dimensions and frame rate
		auto pvi = (VIDEOINFOHEADER2*)m_pInput->CurrentMediaType().Format();
		m_rtAvgSourceFrameTime = pvi->AvgTimePerFrame;
		m_iDimX = pvi->bmiHeader.biWidth;
		m_iDimY = pvi->bmiHeader.biHeight;

		// Initialize the Optical Flow Calculator
		if (m_bP010Input) {
			m_pofcOpticalFlowCalc = new OpticalFlowCalcHDR(m_iDimY, m_iDimX, m_dDimScalar, m_dResolutionDivider);
		} else {
			m_pofcOpticalFlowCalc = new OpticalFlowCalcSDR(m_iDimY, m_iDimX, m_dDimScalar, m_dResolutionDivider);
		}
	}
	return __super::CompleteConnect(dir, pReceivePin);
}


// Transforms an input sample to produce an output sample
HRESULT CHopperRender::Transform(IMediaSample* pIn, IMediaSample* pOut) {
	CheckPointer(pIn, E_POINTER)
	CheckPointer(pOut, E_POINTER)

	// Increment the frame counter
	m_iFrameCounter++;

	// Copy the properties across
	HRESULT hr = DeliverToRenderer(pIn, pOut, FT_TARGET);
	if (FAILED(hr)) {
		return hr;
	}

	return NOERROR;
}


// Called when a new segment is started
HRESULT CHopperRender::NewSegment(REFERENCE_TIME tStart, REFERENCE_TIME tStop, double dRate) {
	// Calculate the current playback frame time
	m_rtCurrPlaybackFrameTime = static_cast<REFERENCE_TIME>(static_cast<double>(m_rtAvgSourceFrameTime) * (1.0 /
		dRate));

	// Check if interpolation is necessary
	if (m_bActivated && m_rtCurrPlaybackFrameTime > FT_TARGET) {
		m_bIntNeeded = true;
	} else {
		m_bIntNeeded = false;
	}

	// Calculate the number of intermediate frames needed
	if (m_bIntNeeded) {
		m_rtCurrStartTime = -1; // Tells the DeliverToRenderer function that we are at the start of a new segment
		m_iNumSamples = static_cast<int>(ceil(
			static_cast<double>(m_rtCurrPlaybackFrameTime) / static_cast<double>(FT_TARGET)));
	} else {
		m_iNumSamples = 1;
	}

	m_bIntTooSlow = false;

	return __super::NewSegment(tStart, tStop, dRate);
}


// Delivers the first sample to the renderer
HRESULT CHopperRender::DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut, REFERENCE_TIME rtAvgFrameTimeTarget) {
	CheckPointer(pIn, E_POINTER)
	CheckPointer(pOut, E_POINTER)

	// Get pointers to the sample buffer
	unsigned char* pInBuffer;
	HRESULT hr = pIn->GetPointer(&pInBuffer);
	if (FAILED(hr)) {
		return hr;
	}

	// Get the size of the output sample
	const long lOutSize = pOut->GetActualDataLength();

	// Get the presentation times for the new output sample
	REFERENCE_TIME rtStartTime, rtEndTime;
	hr = pIn->GetTime(&rtStartTime, &rtEndTime);
	if (FAILED(hr)) {
		return hr;
	}

	// Reset our frame time if necessary and calculate the current number of intermediate frames needed
	if (m_bIntNeeded) {
		//DebugMessage("ACINT | Start time: " + std::to_string(rtStartTime) + " End time: " + std::to_string(rtEndTime) + " Delta: " + std::to_string(rtEndTime - rtStartTime) + " Start2-Start1: " + std::to_string(rtStartTime - m_rtLastStartTime) + " AVG S2-S1: " + std::to_string(m_rtCurrPlaybackFrameTime));
		if (m_rtCurrStartTime == -1) {
			// We are at the start of a new segment
			m_rtCurrStartTime = rtStartTime;
		} else {
			// We are in the middle of a segment
			m_iNumSamples = static_cast<int>(ceil(
				static_cast<double>((m_rtCurrPlaybackFrameTime + rtStartTime - m_rtCurrStartTime)) / static_cast<double>
				(rtAvgFrameTimeTarget)));
		}
		m_rtLastStartTime = rtStartTime;
	}

	// Assemble the output samples
	IMediaSample* pOutNew;
	unsigned char* pOutNewBuffer;

	for (int iIntFrameNum = 0; iIntFrameNum < m_iNumSamples; ++iIntFrameNum) {
		// Create a new output sample
		if (iIntFrameNum < (m_iNumSamples - 1)) {
			pOutNew = nullptr;
			hr = m_pOutput->GetDeliveryBuffer(&pOutNew, nullptr, nullptr, 0);
			if (FAILED(hr)) {
				return hr;
			}
			// Use the input sample for the last output sample
		} else {
			pOutNew = pOut;
		}

		// Get the buffer pointer for the new output sample
		hr = pOutNew->GetPointer(&pOutNewBuffer);
		if (FAILED(hr)) {
			return hr;
		}

		if (m_bIntNeeded) {
			// Interpolation needed
			// Set the new start and end times
			rtStartTime = m_rtCurrStartTime;
			rtEndTime = rtStartTime + rtAvgFrameTimeTarget;
		} else {
			// No interpolation needed
			//DebugMessage("NOINT | Start time: " + std::to_string(rtStartTime) + " End time: " + std::to_string(rtEndTime) + " Delta: " + std::to_string(rtEndTime - rtStartTime) + " Start2-Start1: " + std::to_string(rtStartTime - m_rtLastStartTime) + " AVG S2-S1: " + std::to_string(m_rtCurrPlaybackFrameTime));
			m_iNumSamples = 1;
			m_rtCurrStartTime = rtStartTime;
			m_rtLastStartTime = rtStartTime;
		}

		// Set the new times for the output sample
		hr = pOutNew->SetTime(&rtStartTime, &rtEndTime);
		if (FAILED(hr)) {
			return hr;
		}

		// Calculate the scalar for the interpolation
		const float fScalar = max(
			min((static_cast<float>(m_rtCurrStartTime) - static_cast<float>(m_rtLastStartTime)) / static_cast<float>(
				m_rtCurrPlaybackFrameTime), static_cast<float>(1.0)), static_cast<float>(0.0));
		//const float dScalar = static_cast<float>(iIntFrameNum) / static_cast<float>(m_iNumSamples);

		// Increment the frame time for the next sample
		m_rtCurrStartTime += rtAvgFrameTimeTarget;

		// Copy the media times
		LONGLONG llMediaStart, llMediaEnd;
		if (NOERROR == pIn->GetMediaTime(&llMediaStart, &llMediaEnd)) {
			hr = pOutNew->SetMediaTime(&llMediaStart, &llMediaEnd);
			if (FAILED(hr)) {
				return hr;
			}
		}

		// Copy the Sync point property
		hr = pIn->IsSyncPoint();
		if (hr == S_OK) {
			hr = pOutNew->SetSyncPoint(TRUE);
			if (FAILED(hr)) {
				return hr;
			}
		} else if (hr == S_FALSE) {
			hr = pOutNew->SetSyncPoint(FALSE);
			if (FAILED(hr)) {
				return hr;
			}
		} else {
			// an unexpected error has occured...
			return E_UNEXPECTED;
		}

		// Copy the media type
		AM_MEDIA_TYPE* pMediaType;
		hr = pOut->GetMediaType(&pMediaType);
		if (FAILED(hr)) {
			return hr;
		}
		hr = pOutNew->SetMediaType(pMediaType);
		if (FAILED(hr)) {
			return hr;
		}
		DeleteMediaType(pMediaType);

		// Copy the preroll property
		hr = pIn->IsPreroll();
		if (hr == S_OK) {
			hr = pOutNew->SetPreroll(TRUE);
			if (FAILED(hr)) {
				return hr;
			}
		} else if (hr == S_FALSE) {
			hr = pOutNew->SetPreroll(FALSE);
			if (FAILED(hr)) {
				return hr;
			}
		} else {
			// an unexpected error has occured...
			return E_UNEXPECTED;
		}

		// Copy the discontinuity property
		hr = pIn->IsDiscontinuity();
		if (hr == S_OK) {
			hr = pOutNew->SetDiscontinuity(TRUE);
			if (FAILED(hr)) {
				return hr;
			}
		} else if (hr == S_FALSE) {
			hr = pOutNew->SetDiscontinuity(FALSE);
			if (FAILED(hr)) {
				return hr;
			}
		} else {
			// an unexpected error has occured...
			return E_UNEXPECTED;
		}

		// Copy the actual data length
		hr = pOutNew->SetActualDataLength(lOutSize);
		if (FAILED(hr)) {
			return hr;
		}

		// Interpolate the frame if necessary
		if (m_bIntNeeded) {
			hr = InterpolateFrame(pInBuffer, pOutNewBuffer, fScalar, iIntFrameNum);
			if (FAILED(hr)) {
				return hr;
			}
		} else {
			CopyFrame(pInBuffer, pOutNewBuffer);
		}

		// Deliver the new output sample downstream
		if (iIntFrameNum < (m_iNumSamples - 1)) {
			hr = m_pOutput->Deliver(pOutNew);
			if (FAILED(hr)) {
				return hr;
			}

			// Release the new output sample to avoid memory leaks
			pOutNew->Release();
		}
	}

	return NOERROR;
}


// Copies the source frame to the output frame
HRESULT CHopperRender::CopyFrame(const unsigned char* pInBuffer, unsigned char* pOutBuffer) const {
	m_pofcOpticalFlowCalc->copyFrame(pInBuffer, pOutBuffer);
	return NOERROR;
}


// Interpolates a frame
HRESULT CHopperRender::InterpolateFrame(const unsigned char* pInBuffer, unsigned char* pOutBuffer, const float fScalar, const int iIntFrameNum) {
	// Note the calculation start time at the start of a new sample
	if (iIntFrameNum == 0) {
		m_tpCurrCalcStart = std::chrono::high_resolution_clock::now();
	}

	// Either fill the A or B frame with the new data, so that
	// we always have the current frame and the previous frame
	if (iIntFrameNum == 0) {
		if (m_bBisNewest) {
			m_pofcOpticalFlowCalc->updateFrame1(pInBuffer);
			m_pofcOpticalFlowCalc->blurFrameArray(m_iFrameBlurKernelSize, m_iFrameOutput == 4);
		} else {
			m_pofcOpticalFlowCalc->updateFrame2(pInBuffer);
			m_pofcOpticalFlowCalc->blurFrameArray(m_iFrameBlurKernelSize, m_iFrameOutput == 4);
		}
		m_bBisNewest = !m_bBisNewest;
	}

	// If this is the very first frame, we can't interpolate
	if (m_bFirstFrame) {
		if (m_iFrameCounter > 1) {
			m_bFirstFrame = false;
		}
		return CopyFrame(pInBuffer, pOutBuffer);
	}

	// Calculate the optical flow in both directions and blur it
	if (iIntFrameNum == 0) {
		// Calculate the optical flow (frame 1 to frame 2)
		if (m_iFrameOutput != 4) {
			m_pofcOpticalFlowCalc->calculateOpticalFlow(m_iNumIterations, m_iNumSteps);
		}

		// Flip the flow array to frame 2 to frame 1
		if (m_iFrameOutput == 1 || m_iFrameOutput == 2) {
			m_pofcOpticalFlowCalc->flipFlow();
		}

		// Blur the flow arrays
		if (m_iFrameOutput != 4) {
			m_pofcOpticalFlowCalc->blurFlowArrays(m_iFlowBlurKernelSize);
		}

		// Draw the flow as an HSV image
		if (m_iFrameOutput == 3) {
			m_pofcOpticalFlowCalc->drawFlowAsHSV(1.0, 1.0);
		}
	}

	// Warp frames
	if (m_iFrameOutput == 0 || m_iFrameOutput == 1) {
		m_pofcOpticalFlowCalc->warpFramesForOutput(fScalar, m_iFrameOutput == 0);
	} else if (m_iFrameOutput == 2) {
		m_pofcOpticalFlowCalc->warpFramesForBlending(fScalar);
	}

	// Blend the frames together
	if (m_iFrameOutput == 2) {
		m_pofcOpticalFlowCalc->blendFrames(fScalar);
	}

	// Download the result to the output buffer
	m_pofcOpticalFlowCalc->m_outputFrame.download(pOutBuffer);

	// Adjust the settings to process everything fast enough
	if (m_iFrameOutput != 4 && iIntFrameNum == (m_iNumSamples - 1)) {
		autoAdjustSettings();
	}

	return NOERROR;
}


#define WRITEOUT(var)  hr = pStream->Write(&var, sizeof(var), NULL); \
               if (FAILED(hr)) return hr;

#define READIN(var)    hr = pStream->Read(&var, sizeof(var), NULL); \
               if (FAILED(hr)) return hr;


// This is the only method of IPersist
STDMETHODIMP CHopperRender::GetClassID(CLSID* pClsid) {
	return CBaseFilter::GetClassID(pClsid);
}


// Overriden to write our state into a stream
HRESULT CHopperRender::ScribbleToStream(IStream* pStream) const {
	HRESULT hr;

	WRITEOUT(m_bActivated)
	WRITEOUT(m_iFrameOutput)
	WRITEOUT(m_iNumIterations)
	WRITEOUT(m_iFrameBlurKernelSize)
	WRITEOUT(m_iFlowBlurKernelSize)
	WRITEOUT(m_bIntNeeded)
	WRITEOUT(m_rtCurrPlaybackFrameTime)
	WRITEOUT(m_iNumSteps)

	return NOERROR;
}


// Likewise overriden to restore our state from a stream
HRESULT CHopperRender::ReadFromStream(IStream* pStream) {
	HRESULT hr;

	READIN(m_bActivated)
	READIN(m_iFrameOutput)
	READIN(m_iNumIterations)
	READIN(m_iFrameBlurKernelSize)
	READIN(m_iFlowBlurKernelSize)
	READIN(m_bIntNeeded)
	READIN(m_rtCurrPlaybackFrameTime)
	READIN(m_iNumSteps)

	return NOERROR;
}


// Returns the clsid's of the property pages we support
STDMETHODIMP CHopperRender::GetPages(CAUUID* pPages) {
	CheckPointer(pPages, E_POINTER)

	pPages->cElems = 1;
	pPages->pElems = static_cast<GUID*>(CoTaskMemAlloc(sizeof(GUID)));
	if (pPages->pElems == nullptr) {
		return E_OUTOFMEMORY;
	}

	*(pPages->pElems) = CLSID_HopperRenderSettings;
	return NOERROR;
}


// Return the current settings selected
STDMETHODIMP CHopperRender::get_Settings(bool* pbActivated, int* piFrameOutput,
										 int* piNumIterations, int* piFrameBlurKernelSize, int* piFlowBlurKernelSize, int* piIntActiveState, 
										 double* pdSourceFPS, int* piNumSteps, int* piDimX,
										 int* piDimY, int* piLowDimX, int* piLowDimY) {
	CAutoLock cAutolock(&m_csHopperRenderLock);
	CheckPointer(pbActivated, E_POINTER)
	CheckPointer(piFrameOutput, E_POINTER)
	CheckPointer(piNumIterations, E_POINTER)
	CheckPointer(piFrameBlurKernelSize, E_POINTER)
	CheckPointer(piFlowBlurKernelSize, E_POINTER)
	CheckPointer(piIntActiveState, E_POINTER)
	CheckPointer(pdSourceFPS, E_POINTER)
	CheckPointer(piNumSteps, E_POINTER)
	CheckPointer(piDimX, E_POINTER)
	CheckPointer(piDimY, E_POINTER)
	CheckPointer(piLowDimX, E_POINTER)
	CheckPointer(piLowDimY, E_POINTER)

	*pbActivated = m_bActivated;
	*piFrameOutput = m_iFrameOutput;
	*piNumIterations = m_iNumIterations;
	*piFrameBlurKernelSize = m_iFrameBlurKernelSize;
	*piFlowBlurKernelSize = m_iFlowBlurKernelSize;
	if (m_bIntTooSlow) {
		*piIntActiveState = 3; // Too slow
	} else if (m_bActivated && m_bIntNeeded) {
		*piIntActiveState = 2; // Active
	} else if (m_bActivated && !m_bIntNeeded) {
		*piIntActiveState = 1; // Not needed
	} else {
		*piIntActiveState = 0; // Deactivated
	}
	*pdSourceFPS = 10000000.0 / static_cast<double>(m_rtCurrPlaybackFrameTime);
	*piNumSteps = m_iNumSteps;
	*piDimX = m_iDimX;
	*piDimY = m_iDimY;
	*piLowDimX = m_pofcOpticalFlowCalc->m_iLowDimX;
	*piLowDimY = m_pofcOpticalFlowCalc->m_iLowDimY;

	return NOERROR;
}


// Set the settings
STDMETHODIMP CHopperRender::put_Settings(bool bActivated, int iFrameOutput, int iNumIterations, int iFrameBlurKernelSize, int iFlowBlurKernelSize) {
	CAutoLock cAutolock(&m_csHopperRenderLock);

	m_bActivated = bActivated;
	m_iFrameOutput = iFrameOutput;
	m_iNumIterations = iNumIterations;
	m_iFrameBlurKernelSize = iFrameBlurKernelSize;
	m_iFlowBlurKernelSize = iFlowBlurKernelSize;

	SetDirty(TRUE);
	return NOERROR;
}

// Adjusts the frame scalar
void CHopperRender::adjustFrameScalar(const double dResolutionDivider) {
	if (dResolutionDivider != m_dResolutionDivider && dResolutionDivider > 0.0 && dResolutionDivider <= 1.0) {
		// Re-Calculate the resolution scalar
		m_dResolutionDivider = dResolutionDivider;
		m_dResolutionScalar = 1.0 / m_dResolutionDivider;

		// Re-Initialize the Optical Flow Calculator
		m_bFirstFrame = true;
		m_iFrameCounter = 0;
		m_bBisNewest = true;
		m_bIntTooSlow = false;
		m_bIntNeeded = true;
		m_dCurrCalcDuration = 0.0;
		m_iNumSteps = MIN_NUM_STEPS;
		m_pofcOpticalFlowCalc->m_dResolutionScalar = m_dResolutionScalar;
		m_pofcOpticalFlowCalc->m_dResolutionDivider = m_dResolutionDivider;
		m_pofcOpticalFlowCalc->m_iLowDimX = static_cast<unsigned int>(static_cast<double>(m_iDimX) * m_dResolutionDivider);
		m_pofcOpticalFlowCalc->m_iLowDimY = static_cast<unsigned int>(static_cast<double>(m_iDimY) * m_dResolutionDivider);
		m_pofcOpticalFlowCalc->m_lowGrid.x = static_cast<int>(fmax(ceil(static_cast<double>(m_pofcOpticalFlowCalc->m_iLowDimX) / static_cast<double>(NUM_THREADS)), 1.0));
		m_pofcOpticalFlowCalc->m_lowGrid.y = static_cast<int>(fmax(ceil(static_cast<double>(m_pofcOpticalFlowCalc->m_iLowDimY) / static_cast<double>(NUM_THREADS)), 1.0));
	}
}

// Adjust settings for optimal performance
void CHopperRender::autoAdjustSettings() {
	m_tpCurrCalcEnd = std::chrono::high_resolution_clock::now();
	m_dCurrCalcDuration = std::chrono::duration<double, std::milli>(m_tpCurrCalcEnd - m_tpCurrCalcStart).count();

	// Get the time we have in between source frames (WE NEED TO STAY BELOW THIS!)
	const double dSourceFrameTimeMS = static_cast<double>(m_rtCurrPlaybackFrameTime) / 10000.0;

	/*
	* Calculation took too long
	*/
	if ((m_dCurrCalcDuration + m_dCurrCalcDuration * 0.05) > dSourceFrameTimeMS) {
		DebugMessage(
			"Calculation took too long     " + std::to_string(m_dCurrCalcDuration) + " ms" + 
			" AVG SFT: " + std::to_string(dSourceFrameTimeMS) + 
			" NumSteps: " + std::to_string(m_iNumSteps), LOG_PERFORMANCE);

		// Decrease the number of steps to reduce calculation time
		if (m_iNumSteps > MIN_NUM_STEPS) {
			m_iNumSteps -= 1;
			return;
		}

		// We can't reduce the number of steps any further, so we reduce the resolution divider instead
		if (m_dResolutionDivider > 0.25) {
			// To avoid unnecessary adjustments, we only adjust the resolution divider if we have been too slow for a while
			m_cNumTimesTooSlow += 1;
			if (m_cNumTimesTooSlow > 10) {
				m_cNumTimesTooSlow = 0;
				adjustFrameScalar(m_dResolutionDivider - 0.25);
			}
			return;
		}

		// Disable Interpolation if we are too slow
		m_bIntNeeded = false;
		m_bIntTooSlow = true;

	/*
	* We have left over capacity
	*/
	} else if ((m_dCurrCalcDuration + m_dCurrCalcDuration * 0.1) < dSourceFrameTimeMS) {
		DebugMessage(
			"Calculation has capacity      " + std::to_string(m_dCurrCalcDuration) + " ms" + 
			" AVG SFT: " + std::to_string(dSourceFrameTimeMS) + 
			" NumSteps: " + std::to_string(m_iNumSteps), LOG_PERFORMANCE);

		// Increase the frame scalar if we have enough leftover capacity
		if (m_iNumSteps >= MIN_NUM_STEPS + 8 && m_dResolutionDivider < 1.0) {
			m_cNumTimesTooSlow = 0;
			adjustFrameScalar(m_dResolutionDivider + 0.25);
			m_iNumSteps = MIN_NUM_STEPS;
		} else {
			m_iNumSteps = min(m_iNumSteps + 1, MAX_NUM_STEPS); // Increase the number of steps to use the leftover capacity
		}

	/*
	* Calculation takes as long as it should
	*/
	} else {
		DebugMessage(
			"Calculation took              " + std::to_string(m_dCurrCalcDuration) + " ms" + 
			" AVG SFT: " + std::to_string(dSourceFrameTimeMS) + 
			" NumSteps: " + std::to_string(m_iNumSteps), LOG_PERFORMANCE);
	}
}

// Loads the settings from the registry
HRESULT CHopperRender::loadSettings() {
	HKEY hKey;
    LPCWSTR subKey = L"SOFTWARE\\HopperRender";
	DWORD value;
	DWORD dataSize = sizeof(DWORD);
	LPCWSTR valueName; 

    // Open the registry key
    LONG result0 = RegOpenKeyEx(HKEY_CURRENT_USER, subKey, 0, KEY_READ, &hKey);
    
    if (result0 == ERROR_SUCCESS) {
        // Load activated state
		valueName = L"Activated"; 
        LONG result1 = RegQueryValueEx(hKey, valueName, NULL, NULL, reinterpret_cast<BYTE*>(&value), &dataSize);
		m_bActivated = value;

		// Load Frame Output
		valueName = L"FrameOutput"; 
        LONG result2 = RegQueryValueEx(hKey, valueName, NULL, NULL, reinterpret_cast<BYTE*>(&value), &dataSize);
		m_iFrameOutput = value;

		// Load the number of iterations
		valueName = L"NumIterations"; 
        LONG result3 = RegQueryValueEx(hKey, valueName, NULL, NULL, reinterpret_cast<BYTE*>(&value), &dataSize);
		m_iNumIterations = value;

		// Load the frame blur kernel size
		valueName = L"FrameBlurKernelSize"; 
        LONG result4 = RegQueryValueEx(hKey, valueName, NULL, NULL, reinterpret_cast<BYTE*>(&value), &dataSize);
		m_iFrameBlurKernelSize = value;

		// Load the flow blur kernel size
		valueName = L"FlowBlurKernelSize"; 
        LONG result5 = RegQueryValueEx(hKey, valueName, NULL, NULL, reinterpret_cast<BYTE*>(&value), &dataSize);
		m_iFlowBlurKernelSize = value;

		RegCloseKey(hKey); // Close the registry key

		// Check for errors
        if (result1 || result2 || result3 || result4) {
			return E_FAIL;
        } else {
            return S_OK;
        }

    } else {
        return E_FAIL;
    }
}