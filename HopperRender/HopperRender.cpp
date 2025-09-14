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

#include "opticalFlowCalcSDR.h"
#include "opticalFlowCalcHDR.h"
#include "resource.h"
#include "CustomInputPin.h"
#include "IMediaSideData.h"
#include <atlcomcli.h>

// Debug message function
void DebugMessage(const std::string& logMessage, const bool showLog) {
	if (showLog) OutputDebugStringA((logMessage + "\n").c_str());
}

DWORD WINAPI MessageBoxThread(LPVOID lpParam) {
    LPCWSTR message = reinterpret_cast<LPCWSTR>(lpParam);
    MessageBox(NULL, message, TEXT("CUDA Error"), MB_OK | MB_ICONERROR);
    return 0;
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
	// Settings
	m_iFrameOutput(BlendedFrame),

	// Video info
	m_iDimX(1),
	m_iDimY(1),

	// Timings
	m_rtCurrStartTime(-1),
	m_rtSourceFrameTime(0),
	m_rtTargetFrameTime(166667),
	m_rtCurrPlaybackFrameTime(0),

	// Optical Flow calculation
	m_pofcOpticalFlowCalc(nullptr),

	// Frame output
	m_iFrameCounter(0),
	m_iNumIntFrames(1),

	// Performance and activation status
	m_iIntActiveState(Active),
	m_dTotalWarpDuration(0.0),
	blendingScalar(0.0)
	{

	m_pInput = new CCustomInputPin(phr, L"XForm In", this);
	m_pOutput = new CTransformOutputPin(NAME("Transform output pin"), this, phr, L"XForm Out");
}

void CHopperRender::useDisplayRefreshRate() {
    DEVMODE devMode;
    ZeroMemory(&devMode, sizeof(devMode));
    devMode.dmSize = sizeof(devMode);

    if (EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &devMode)) {
	DWORD refreshRate = devMode.dmDisplayFrequency;
	m_rtTargetFrameTime = (1.0 / (double)refreshRate) * 1e7;
    }
}

template <class T> void SafeRelease(T** ppT) {
    if (*ppT) {
	(*ppT)->Release();
	*ppT = nullptr;
    }
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

	// We only accept the VideoInfo2 header
	if (*mtIn->FormatType() != FORMAT_VideoInfo2) {
		return E_INVALIDARG;
	}

	// We only accept NV12 or P010 input
	if (IsEqualGUID(*mtIn->Type(), MEDIATYPE_Video) && 
		(IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_NV12) || IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_P010))) {
		return NOERROR;
	}
	return E_FAIL;
}

// Checks whether an input media type is compatible with an output media type
HRESULT CHopperRender::CheckTransform(const CMediaType* mtIn, const CMediaType* mtOut) {
	CheckPointer(mtIn, E_POINTER)
	CheckPointer(mtOut, E_POINTER)

	// We can transform NV12 or P010 input to P010 output
	if (IsEqualGUID(*mtIn->Type(), MEDIATYPE_Video) && 
		(IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_NV12) || IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_P010)) && 
		IsEqualGUID(*mtOut->Subtype(), MEDIASUBTYPE_P010)) {
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

	ppropInputRequest->cBuffers = 5;
	ppropInputRequest->cbBuffer = m_pOutput->CurrentMediaType().GetSampleSize();
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

	ASSERT(Actual.cBuffers == 5);

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
HRESULT CHopperRender::UpdateVideoInfoHeader(CMediaType* pMediaType) {
	// Get the input media type information
	CMediaType* mtIn = &m_pInput->CurrentMediaType();
	auto pvi = (VIDEOINFOHEADER2*)mtIn->Format();
	const long biWidth = pvi->bmiHeader.biWidth;
	const long biHeight = pvi->bmiHeader.biHeight;
	const unsigned int dwX = pvi->dwPictAspectRatioX;
	const unsigned int dwY = pvi->dwPictAspectRatioY;
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
	vih2->dwInterlaceFlags = 0;
	vih2->dwControlFlags = pvi->dwControlFlags;
	vih2->dwReserved1 = pvi->dwReserved1;
	vih2->dwReserved2 = pvi->dwReserved2;

	// Set the BitmapInfoHeader information
	BITMAPINFOHEADER* pBIH = nullptr;
	pBIH = &vih2->bmiHeader;
	pBIH->biSize = sizeof(BITMAPINFOHEADER);
	pBIH->biWidth = biWidth;
	pBIH->biHeight = biHeight;
	pBIH->biBitCount = 24;
	pBIH->biPlanes = 1;                                      // VERIFY
	pBIH->biSizeImage = (biWidth * biHeight * 24) >> 3;      // VERIFY
	pBIH->biCompression = guid.Data1;

	// Set the media type information
	pMediaType->SetSampleSize((biWidth * biHeight * 24) >> 3);
	pMediaType->SetTemporalCompression(0);

	return NOERROR;
}

// Retrieves a filter from a pin
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
	// Check if we're connecting to the output pin
	if (dir == PINDIR_OUTPUT) {
		// Get the frame dimensions and frame rate
		auto pvi = (VIDEOINFOHEADER2*)m_pInput->CurrentMediaType().Format();
		m_rtSourceFrameTime = pvi->AvgTimePerFrame;
		m_iDimX = pvi->bmiHeader.biWidth;
		m_iDimY = pvi->bmiHeader.biHeight;
	}
	return __super::CompleteConnect(dir, pReceivePin);
}

// Called when a new sample (source frame) arrives
HRESULT CHopperRender::Transform(IMediaSample* pIn, IMediaSample* pOut) {
	CheckPointer(pIn, E_POINTER)
	CheckPointer(pOut, E_POINTER)

	// Update the output media type with potentially new video info from the input pin
	if (m_iFrameCounter == 0) {
		CMediaType* pMediaType = &m_pOutput->CurrentMediaType();
		UpdateVideoInfoHeader(pMediaType);
		m_pOutput->SetMediaType(pMediaType);
	}

	m_iFrameCounter++;

	HRESULT hr = DeliverToRenderer(pIn, pOut);
	if (FAILED(hr)) {
		return hr;
	}

	return NOERROR;
}

void CHopperRender::UpdateInterpolationStatus() {
	// Check if interpolation is necessary
	if (m_iIntActiveState && m_rtCurrPlaybackFrameTime > m_rtTargetFrameTime) {
		m_iIntActiveState = Active;
	} else if (m_iIntActiveState) {
		m_iIntActiveState = NotNeeded;
	}

	m_iFrameCounter = 0;
	m_rtCurrStartTime = -1; // Tells the DeliverToRenderer function that we are at the start of a new segment
}

// Called when a new segment is started (by seeking or changing the playback speed)
HRESULT CHopperRender::NewSegment(REFERENCE_TIME tStart, REFERENCE_TIME tStop, double dRate) {
	// Calculate the current source playback frame time
	m_rtCurrPlaybackFrameTime = static_cast<REFERENCE_TIME>(static_cast<double>(m_rtSourceFrameTime) * (1.0 / dRate));

	UpdateInterpolationStatus();

	return __super::NewSegment(tStart, tStop, dRate);
}

// Delivers the new samples (interpolated frames) to the renderer
HRESULT CHopperRender::DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut) {
	CheckPointer(pIn, E_POINTER)
	CheckPointer(pOut, E_POINTER)

	// Get pointers to the sample buffer
	unsigned char* pInBuffer;
	HRESULT hr = pIn->GetPointer(&pInBuffer);
	if (FAILED(hr)) {
		return hr;
	}

	// Get the side data
	const BYTE* sideDataBytes1;
	const BYTE* sideDataBytes2;
	const BYTE* sideDataBytes3;
	const BYTE* sideDataBytes4;
	const BYTE* sideDataBytes5;
	const BYTE* sideDataBytes6;
	const BYTE* sideDataBytes7;
	const BYTE* sideDataBytes8;
	size_t sideDataSize1 = 0;
	size_t sideDataSize2 = 0;
	size_t sideDataSize3 = 0;
	size_t sideDataSize4 = 0;
	size_t sideDataSize5 = 0;
	size_t sideDataSize6 = 0;
	size_t sideDataSize7 = 0;
	size_t sideDataSize8 = 0;
	if (CComQIPtr<IMediaSideData> sideDataIn = pIn) {
		hr = sideDataIn->GetSideData(IID_MediaSideDataDOVIMetadata,
					     &sideDataBytes1, &sideDataSize1);
		hr = sideDataIn->GetSideData(IID_MediaSideDataDOVIRPU,
					     &sideDataBytes2, &sideDataSize2);
		hr = sideDataIn->GetSideData(IID_MediaSideDataControlFlags,
					     &sideDataBytes3, &sideDataSize3);
		hr = sideDataIn->GetSideData(IID_MediaSideDataHDR,
					     &sideDataBytes4, &sideDataSize4);
		hr = sideDataIn->GetSideData(IID_MediaSideDataHDR10Plus,
					     &sideDataBytes5, &sideDataSize5);
		hr = sideDataIn->GetSideData(IID_MediaSideDataHDRContentLightLevel,
						 &sideDataBytes6, &sideDataSize6);
		hr = sideDataIn->GetSideData(IID_MediaSideDataEIA608CC,
					     &sideDataBytes7, &sideDataSize7);
		hr = sideDataIn->GetSideData(IID_MediaSideData3DOffset,
					     &sideDataBytes8, &sideDataSize8);
	}

	// Get the size of the output sample
	const long lOutSize = pOut->GetActualDataLength();

	// Initialize the Optical Flow Calculator
	if (m_pofcOpticalFlowCalc == nullptr) {
	    const long outputFrameWidth = lOutSize / (m_iDimY * 3);
	    int deltaScalar;
	    int neighborScalar;
	    float blackLevel;
	    float whiteLevel;
	    int customResScalar;
	    loadSettings(&deltaScalar, &neighborScalar, &blackLevel, &whiteLevel, &customResScalar);
	    if (sideDataSize4 > 0 || m_pInput->CurrentMediaType().subtype == MEDIASUBTYPE_P010) {
			m_pofcOpticalFlowCalc = new OpticalFlowCalcHDR(m_iDimY, outputFrameWidth, m_iDimX, deltaScalar, neighborScalar, blackLevel, whiteLevel, customResScalar);
		} else {
			m_pofcOpticalFlowCalc = new OpticalFlowCalcSDR(m_iDimY, outputFrameWidth, m_iDimX, deltaScalar, neighborScalar, blackLevel, whiteLevel, customResScalar);
		}
	}

	// Get the presentation times for the new output sample
	REFERENCE_TIME rtStartTime, rtEndTime;
	hr = pIn->GetTime(&rtStartTime, &rtEndTime);
	if (FAILED(hr)) {
		return hr;
	}

	// Reset our frame time if necessary and calculate the current number of intermediate frames needed
	if (m_rtCurrStartTime == -1) {
		// We are at the start of a new segment
		m_rtCurrStartTime = rtStartTime;
	}

	// Calculate the number of interpolated frames
	if (m_iIntActiveState == Active) {
		m_iNumIntFrames = (int)max(ceil((1.0 - blendingScalar) / ((double)m_rtTargetFrameTime / (double)m_rtCurrPlaybackFrameTime)), 1.0);
	} else {
		m_iNumIntFrames = 1;
	}

	// Adjust the settings to process everything fast enough
	autoAdjustSettings();

	m_pofcOpticalFlowCalc->updateFrame(pInBuffer);

	if (m_iIntActiveState == Active && m_iFrameCounter >= 2) {
		// Calculate the optical flow (frame 1 to frame 2)
		m_pofcOpticalFlowCalc->calculateOpticalFlow();
	}

	// Assemble the output samples
	IMediaSample* pOutNew;
	unsigned char* pOutNewBuffer;

	for (int iIntFrameNum = 0; iIntFrameNum < m_iNumIntFrames; ++iIntFrameNum) {
		// Create a new output sample
		if (iIntFrameNum < (m_iNumIntFrames - 1)) {
			pOutNew = nullptr;
			hr = m_pOutput->GetDeliveryBuffer(&pOutNew, nullptr, nullptr, 0);
			if (FAILED(hr)) {
				return hr;
			}
			// Use the input sample for the last output sample
		} else {
			pOutNew = pOut;
		}

		// Set the side data
		IMediaSideData* sideDataOut;
		if (SUCCEEDED(hr = pOutNew->QueryInterface(&sideDataOut)))
		{
		    if (sideDataSize1 > 0) {
				hr = sideDataOut->SetSideData(IID_MediaSideDataDOVIMetadata, sideDataBytes1, sideDataSize1);
			}
			if (sideDataSize2 > 0) {
				hr = sideDataOut->SetSideData(IID_MediaSideDataDOVIRPU, sideDataBytes2, sideDataSize2);
			}
			if (sideDataSize3 > 0) {
				hr = sideDataOut->SetSideData(IID_MediaSideDataControlFlags, sideDataBytes3, sideDataSize3);
			}
			if (sideDataSize4 > 0) {
				hr = sideDataOut->SetSideData(IID_MediaSideDataHDR, sideDataBytes4, sideDataSize4);
			}
			if (sideDataSize5 > 0) {
				hr = sideDataOut->SetSideData(IID_MediaSideDataHDR10Plus, sideDataBytes5, sideDataSize5);
			}
			if (sideDataSize6 > 0) {
				hr = sideDataOut->SetSideData(IID_MediaSideDataHDRContentLightLevel, sideDataBytes6, sideDataSize6);
			}
			if (sideDataSize7 > 0) {
				hr = sideDataOut->SetSideData(IID_MediaSideDataEIA608CC, sideDataBytes7, sideDataSize7);
			}
			if (sideDataSize8 > 0) {
				hr = sideDataOut->SetSideData(IID_MediaSideData3DOffset, sideDataBytes8, sideDataSize8);
			}

		    SafeRelease(&sideDataOut);
		}

		// Get the buffer pointer for the new output sample
		hr = pOutNew->GetPointer(&pOutNewBuffer);
		if (FAILED(hr)) {
			return hr;
		}

		// Set the new start and end times
		if (m_iIntActiveState == Active) {
		    rtStartTime = m_rtCurrStartTime;
		    rtEndTime = rtStartTime + m_rtTargetFrameTime;
		}

		// Set the new times for the output sample
		hr = pOutNew->SetTime(&rtStartTime, &rtEndTime);
		if (FAILED(hr)) {
			return hr;
		}

		// Increment the frame time for the next sample
		m_rtCurrStartTime += m_rtTargetFrameTime;

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
			return E_UNEXPECTED;
		}

		// Copy the actual data length
		hr = pOutNew->SetActualDataLength(lOutSize);
		if (FAILED(hr)) {
			return hr;
		}

		// Interpolate the frame if necessary
		if (m_iIntActiveState == Active && m_iFrameCounter >= 2) {
			m_pofcOpticalFlowCalc->warpFrames(blendingScalar, m_iFrameOutput);
		} else {
			m_pofcOpticalFlowCalc->copyFrame();
		}

		// Download the result to the output buffer
		m_pofcOpticalFlowCalc->downloadFrame(pOutNewBuffer);

		// Retrieve how long the warp calculation took
		m_dTotalWarpDuration += m_pofcOpticalFlowCalc->m_warpCalcTime;

		// Increase the blending scalar
		if (m_iIntActiveState == Active) {
			blendingScalar += (double)m_rtTargetFrameTime / (double)m_rtCurrPlaybackFrameTime;
			if (blendingScalar >= 1.0) {
				blendingScalar -= 1.0;
			}
		}

		// Deliver the new output sample downstream
		// We don't need to deliver the last sample, as it is automatically delivered by the caller
		if (iIntFrameNum < (m_iNumIntFrames - 1)) {
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

#define WRITEOUT(var)  hr = pStream->Write(&var, sizeof(var), NULL); \
               if (FAILED(hr)) return hr;

#define READIN(var)    hr = pStream->Read(&var, sizeof(var), NULL); \
               if (FAILED(hr)) return hr;

// This is the only method of IPersist
STDMETHODIMP CHopperRender::GetClassID(CLSID* pClsid) {
	return CBaseFilter::GetClassID(pClsid);
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
STDMETHODIMP CHopperRender::GetCurrentSettings(bool* pbActivated,
		int* piFrameOutput,
		double* pdTargetFPS,
		int* piDeltaScalar,
		int* piNeighborScalar,
		int* piBlackLevel,
		int* piWhiteLevel,
		int* piIntActiveState,
		double* pdSourceFPS,
		double* pdOFCCalcTime,
		double* pdWarpCalcTime,
		int* piDimX,
		int* piDimY,
		int* piLowDimX,
		int* piLowDimY) {
	CAutoLock cAutolock(&m_csHopperRenderLock);
	CheckPointer(pbActivated, E_POINTER)
	CheckPointer(piFrameOutput, E_POINTER)
	CheckPointer(pdTargetFPS, E_POINTER)
	CheckPointer(piDeltaScalar, E_POINTER)
	CheckPointer(piNeighborScalar, E_POINTER)
	CheckPointer(piBlackLevel, E_POINTER)
	CheckPointer(piWhiteLevel, E_POINTER)
	CheckPointer(piIntActiveState, E_POINTER)
	CheckPointer(pdSourceFPS, E_POINTER)
	CheckPointer(pdOFCCalcTime, E_POINTER)
	CheckPointer(pdWarpCalcTime, E_POINTER)
	CheckPointer(piDimX, E_POINTER)
	CheckPointer(piDimY, E_POINTER)
	CheckPointer(piLowDimX, E_POINTER)
	CheckPointer(piLowDimY, E_POINTER)

	if (m_pofcOpticalFlowCalc == nullptr) {
		// Optical Flow Calculator not initialized yet
		int deltaScalar;
	    int neighborScalar;
	    float blackLevel;
	    float whiteLevel;
	    int customResScalar;
	    loadSettings(&deltaScalar, &neighborScalar, &blackLevel, &whiteLevel, &customResScalar);
		*pbActivated = m_iIntActiveState != 0;
		*piFrameOutput = m_iFrameOutput;
		if (*pdTargetFPS == 0.0) {
			*pdTargetFPS = 10000000.0 / static_cast<double>(m_rtTargetFrameTime);
		}
		*piDeltaScalar = deltaScalar;
		*piNeighborScalar = neighborScalar;
		*piBlackLevel = blackLevel / 256.0f;
		*piWhiteLevel = whiteLevel / 256.0f;
		*piIntActiveState = Active;
		*pdSourceFPS = 0.0;
		*pdOFCCalcTime = 0.0;
		*pdWarpCalcTime = 0.0;
		*piDimX = 0;
		*piDimY = 0;
		*piLowDimX = 0;
		*piLowDimY = 0;
	} else {
		*pbActivated = m_iIntActiveState != 0;
		*piFrameOutput = m_iFrameOutput;
		if (*pdTargetFPS == 0.0) {
			*pdTargetFPS = 10000000.0 / static_cast<double>(m_rtTargetFrameTime);
		}
		*piDeltaScalar = m_pofcOpticalFlowCalc->m_deltaScalar;
		*piNeighborScalar = m_pofcOpticalFlowCalc->m_neighborBiasScalar;
		*piBlackLevel = (int)(m_pofcOpticalFlowCalc->m_outputBlackLevel) >> 8;
		*piWhiteLevel = (int)(m_pofcOpticalFlowCalc->m_outputWhiteLevel) >> 8;
		*piIntActiveState = m_iIntActiveState;
		*pdSourceFPS = 10000000.0 / static_cast<double>(m_rtCurrPlaybackFrameTime);
		*pdOFCCalcTime = 1000.0 * m_pofcOpticalFlowCalc->m_ofcCalcTime;
		*pdWarpCalcTime = 1000.0 * m_dTotalWarpDuration;
		*piDimX = m_iDimX;
		*piDimY = m_iDimY;
		*piLowDimX = m_pofcOpticalFlowCalc->m_opticalFlowFrameWidth;
		*piLowDimY = m_pofcOpticalFlowCalc->m_opticalFlowFrameHeight;
	}
	return NOERROR;
}

// Apply the new settings
STDMETHODIMP CHopperRender::UpdateUserSettings(bool bActivated, int iFrameOutput, double dTargetFPS, int iDeltaScalar, int iNeighborScalar, int iBlackLevel, int iWhiteLevel) {
	CAutoLock cAutolock(&m_csHopperRenderLock);

	if (!bActivated) {
		m_iIntActiveState = Deactivated;
	} else if (!m_iIntActiveState) {
		m_iIntActiveState = Active;
	}
	m_iFrameOutput = static_cast<FrameOutput>(iFrameOutput);
	if (dTargetFPS > 0.0) {
	    m_rtTargetFrameTime = (1.0 / (double)dTargetFPS) * 1e7;
	} else {
	    useDisplayRefreshRate();
	}
	UpdateInterpolationStatus();
	if (m_pofcOpticalFlowCalc != nullptr) {
		m_pofcOpticalFlowCalc->m_deltaScalar = iDeltaScalar;
		m_pofcOpticalFlowCalc->m_neighborBiasScalar = iNeighborScalar;
		m_pofcOpticalFlowCalc->m_outputBlackLevel = (float)(iBlackLevel << 8);
		m_pofcOpticalFlowCalc->m_outputWhiteLevel = (float)(iWhiteLevel << 8);
	}

	return NOERROR;
}

// Adjust settings for optimal performance
void CHopperRender::autoAdjustSettings() {
	// Get the time we have in between source frames (WE NEED TO STAY BELOW THIS!)
	const double dSourceFrameTimeMS = static_cast<double>(m_rtCurrPlaybackFrameTime) / 10000000.0;

	double currMaxCalcDuration = m_pofcOpticalFlowCalc->m_ofcCalcTime + m_dTotalWarpDuration;

    // Check if we were too slow or have leftover capacity
    if ((currMaxCalcDuration * UPPER_PERF_BUFFER) > dSourceFrameTimeMS) {
        if (m_pofcOpticalFlowCalc->m_opticalFlowSearchRadius > MIN_SEARCH_RADIUS) {
            // Decrease the number of steps to reduce calculation time
            m_pofcOpticalFlowCalc->m_opticalFlowSearchRadius--;
        } else {
            // Disable Interpolation if we are too slow
            //m_iIntActiveState = TooSlow;
        }

    } else if ((currMaxCalcDuration * LOWER_PERF_BUFFER) < dSourceFrameTimeMS) {
        // Increase the frame scalar if we have enough leftover capacity
        if (m_pofcOpticalFlowCalc->m_opticalFlowSearchRadius < MAX_SEARCH_RADIUS) {
            m_pofcOpticalFlowCalc->m_opticalFlowSearchRadius++;
        }
    }

    // Reset the warp duration for the next frame
    m_dTotalWarpDuration = 0.0;
}

// Loads the settings from the registry
HRESULT CHopperRender::loadSettings(int* deltaScalar, int* neighborScalar,
				    float* blackLevel, float* whiteLevel,
				    int* maxCalcRes) {
    HKEY hKey;
    LPCWSTR subKey = L"SOFTWARE\\HopperRender";
    DWORD value = 0;
    double value2 = 0.0;
    DWORD dataSize = sizeof(DWORD);
    DWORD dataSize2 = sizeof(double);
    LPCWSTR valueName;
    LONG result;

    // Open the registry key
    result = RegOpenKeyEx(HKEY_CURRENT_USER, subKey, 0, KEY_READ, &hKey);

    if (result == ERROR_SUCCESS) {
		// Load activated state
		valueName = L"Activated";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0) {
			if (value)
				m_iIntActiveState = Active;
			else
				m_iIntActiveState = Deactivated;
		} else {
			m_iIntActiveState = Active;
		}

		// Load Frame Output
		valueName = L"FrameOutput";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0) {
		    m_iFrameOutput = static_cast<FrameOutput>(value);
		} else {
			m_iFrameOutput = BlendedFrame;
		}

		// Load the target fps
		valueName = L"TargetFPS";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value2), &dataSize2);
		if (value2 > 0 && result == 0) {
			m_rtTargetFrameTime = (1.0 / (double)value2) * 1e7;
		} else {
			useDisplayRefreshRate();
		}

		// Load the delta scalar
		valueName = L"DeltaScalar";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0 && value <= 10) {
			*deltaScalar = value;
		} else {
			*deltaScalar = 8;
		}

		// Load the neighbor scalar
		valueName = L"NeighborScalar";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0 && value <= 10) {
			*neighborScalar = value;
		} else {
			*neighborScalar = 6;
		}

		// Load the black level
		valueName = L"BlackLevel";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0 && value <= 255) {
			*blackLevel = (float)(value << 8);
		} else {
			*blackLevel = 0.0f;
		}

		// Load the white level
		valueName = L"WhiteLevel";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0 && value <= 255) {
			*whiteLevel = (float)(value << 8);
		} else {
			*whiteLevel = 65535.0f;
		}

		// Load the maximum calculation resolution
		valueName = L"MaxCalcRes";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 32) {
			*maxCalcRes = value;
		} else {
			*maxCalcRes = MAX_CALC_RES;
		}

		RegCloseKey(hKey); // Close the registry key

    } else {
		// Load the default values
		m_iIntActiveState = Active;
		m_iFrameOutput = BlendedFrame;
		useDisplayRefreshRate();
		*deltaScalar = DEFAULT_DELTA_SCALAR;
		*neighborScalar = DEFAULT_NEIGHBOR_SCALAR;
		*blackLevel = DEFAULT_BLACK_LEVEL;
		*whiteLevel = DEFAULT_WHITE_LEVEL << 8;
		*maxCalcRes = MAX_CALC_RES;
    }
    UpdateInterpolationStatus();
    return NOERROR;
}