#include <windows.h>
#include <streams.h>
#include <string>
#include <initguid.h>
#include <ctime>
#include <iomanip>
#include <sstream>

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
#include "version.h"
#include "CustomInputPin.h"
#include "IMediaSideData.h"
#include <atlcomcli.h>
#include <vector>
#include <cwchar>

// Input pin types
constexpr AMOVIESETUP_MEDIATYPE sudPinTypesIn[] = 
{
	{
		&MEDIATYPE_Video, // Major type
		&MEDIASUBTYPE_NV12 // Minor type
	},
	{
		&MEDIATYPE_Video, // Major type
		&MEDIASUBTYPE_P010 // Minor type
	}
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
		2, // Number of types
		sudPinTypesIn // Pin information
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

BOOL APIENTRY DllMain(HANDLE hModule, DWORD dwReason, LPVOID lpReserved) {
    return DllEntryPoint(static_cast<HINSTANCE>(hModule), dwReason, lpReserved);
}

// Logging method
void CHopperRender::Log(LogLevel level, const char* functionName, const char* message) {
    CAutoLock lock(&m_csHopperRenderLock);
    
    if (!m_logFile.is_open()) return;
    
    std::time_t now = std::time(nullptr);
    std::tm timeInfo;
    localtime_s(&timeInfo, &now);
    
    const char* levelStr = (level == LogLevel::Error) ? "ERROR" : "INFO";
    m_logFile << "[" << std::put_time(&timeInfo, "%H:%M:%S") << "] "
              << levelStr << " in " << functionName << ": " << message << "\n";
    m_logFile.flush();
    
    // Also output to debug console
    std::ostringstream debugMsg;
    debugMsg << "HopperRender " << levelStr << " in " << functionName << ": " << message;
    OutputDebugStringA(debugMsg.str().c_str());
}

// Constructor
CHopperRender::CHopperRender(TCHAR* tszName, LPUNKNOWN punk, HRESULT* phr) : CTransformFilter(tszName, punk, CLSID_HopperRender),
    // Settings
    m_iFrameOutput(BlendedFrame),

    // Video info
	m_iDimX(1),
	m_iDimY(1),

    // Timings
	m_rtCurrStartTime(-1),
	m_rtSourceFrameTime(417083),
	m_rtTargetFrameTime(166667),
	m_rtCurrPlaybackFrameTime(417083),

    // Optical Flow calculation
    m_pofcOpticalFlowCalc(nullptr),

    // Frame output
	m_iFrameCounter(0),
	m_iNumIntFrames(1),

	// Performance and activation status
	m_iIntActiveState(Active),
	m_dTotalWarpDuration(0.0),
	m_dBlendingScalar(0.0),
    m_bUseDisplayFPS(true),
	m_bValidFrameTimes(false),
	m_iSceneChangeThreshold(DEFAULT_SCENE_CHANGE_THRESHOLD),
	m_iPeakTotalFrameDelta(0),
	m_iBufferFrames(DEFAULT_BUFFER_FRAMES)
	{
    // Initialize logging
    char tempPath[MAX_PATH];
    GetTempPathA(MAX_PATH, tempPath);
    std::time_t now = std::time(nullptr);
    std::tm timeInfo;
    localtime_s(&timeInfo, &now);
    std::ostringstream logFileName;
    logFileName << tempPath << "HopperRender_" << std::put_time(&timeInfo, "%Y%m%d_%H%M%S") << ".log";
    m_logFile.open(logFileName.str(), std::ios::out | std::ios::app);
    
    if (m_logFile.is_open()) {
        m_logFile << "[" << std::put_time(&timeInfo, "%H:%M:%S") << "] " VERSION_STRING_WITH_NAME " - Filter instance created\n";
        m_logFile.flush();
    }

    UpdateInterpolationStatus();

    delete m_pInput;
    delete m_pOutput;
    m_pInput = new CCustomInputPin(phr, L"XForm In", this);
	m_pOutput = new CTransformOutputPin(NAME("Transform output pin"), this, phr, L"XForm Out");
}

// Destructor
CHopperRender::~CHopperRender() {
	Log(LogLevel::Info, "~CHopperRender", "Cleaning up filter instance");
	
	if (m_pofcOpticalFlowCalc) {
		delete m_pofcOpticalFlowCalc;
		m_pofcOpticalFlowCalc = nullptr;
	}
	
	if (m_logFile.is_open()) {
		m_logFile.close();
	}
}

// Override Stop to ensure proper cleanup
STDMETHODIMP CHopperRender::Stop() {
    Log(LogLevel::Info, "Stop", "Filter stopping");
    
    HRESULT hr = CTransformFilter::Stop();
    
    if (FAILED(hr)) {
        Log(LogLevel::Error, "Stop", "Failed to stop filter");
    } else {
        Log(LogLevel::Info, "Stop", "Filter stopped successfully");
    }
    
    return hr;
}

static inline double ComputeRefreshHz(const DISPLAYCONFIG_PATH_INFO& path, const DISPLAYCONFIG_MODE_INFO* modes) noexcept {
    double freq = 0.0;

    if (path.targetInfo.modeInfoIdx != DISPLAYCONFIG_PATH_MODE_IDX_INVALID) {
        const DISPLAYCONFIG_MODE_INFO* mode = &modes[path.targetInfo.modeInfoIdx];
        if (mode->infoType == DISPLAYCONFIG_MODE_INFO_TYPE_TARGET) {
            const DISPLAYCONFIG_RATIONAL* vSyncFreq = &mode->targetMode.targetVideoSignalInfo.vSyncFreq;
            if (vSyncFreq->Denominator != 0 && vSyncFreq->Numerator / vSyncFreq->Denominator > 1) {
                freq = static_cast<double>(vSyncFreq->Numerator) / static_cast<double>(vSyncFreq->Denominator);
            }
        }
    }

    if (freq == 0.0) {
        const DISPLAYCONFIG_RATIONAL* refreshRate = &path.targetInfo.refreshRate;
        if (refreshRate->Denominator != 0 && refreshRate->Numerator / refreshRate->Denominator > 1) {
            freq = static_cast<double>(refreshRate->Numerator) / static_cast<double>(refreshRate->Denominator);
        }
    }

    return freq;
}

double GetDisplayRefreshRateByName(const wchar_t* displayName) noexcept {
    if (!displayName || !displayName[0]) {
        return 0.0;
    }

    UINT32 numPaths = 0;
    UINT32 numModes = 0;
    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
    LONG res;

    // The display configuration may change; loop until buffers are sized correctly.
    do {
        res = GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &numPaths, &numModes);
        if (res == ERROR_SUCCESS) {
            paths.resize(numPaths);
            modes.resize(numModes);
            res = QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &numPaths, paths.data(), &numModes, modes.data(), nullptr);
        }
    } while (res == ERROR_INSUFFICIENT_BUFFER);

    if (res != ERROR_SUCCESS) {
        return 0.0;
    }

    // numPaths/numModes could have decreased between calls
    paths.resize(numPaths);
    modes.resize(numModes);

    for (const auto& path : paths) {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source = {
            { DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(source), path.sourceInfo.adapterId, path.sourceInfo.id }, {}
        };

        if (DisplayConfigGetDeviceInfo(&source.header) == ERROR_SUCCESS) {
            if (wcscmp(displayName, source.viewGdiDeviceName) == 0) {
                return ComputeRefreshHz(path, modes.data());
            }
        }
    }

    return 0.0;
}

double GetDisplayRefreshRateForWindow(HWND hwnd) noexcept {
    HMONITOR hMon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);
    if (!hMon) {
        return 0.0;
    }

    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(hMon, reinterpret_cast<MONITORINFO*>(&mi))) {
        return 0.0;
    }

    // mi.szDevice is the GDI device name, e.g. "\\.\DISPLAY1"
    return GetDisplayRefreshRateByName(mi.szDevice);
}

void CHopperRender::useDisplayRefreshRate() {
	// Get the window handle of the renderer
    HWND hwnd = nullptr;
    if (m_pGraph) {
        CComPtr<IVideoWindow> pVW;
        if (SUCCEEDED(m_pGraph->QueryInterface(IID_IVideoWindow, (void**)&pVW)) && pVW) {
            OAHWND oahwnd = 0;
            if (SUCCEEDED(pVW->get_Owner(&oahwnd))) {
                hwnd = reinterpret_cast<HWND>(oahwnd);
            }
        }
    }

	// Get the display refresh rate of the monitor the window is on
    double refreshRate = GetDisplayRefreshRateForWindow(hwnd);
	m_rtTargetFrameTime = (1.0 / (double)refreshRate) * 1e7;
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

    if (pNewObject == nullptr && phr) {
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

	// We only accept the VideoInfo & VideoInfo2 format types
	GUID formatType = *mtIn->FormatType();
	if (formatType != FORMAT_VideoInfo && formatType != FORMAT_VideoInfo2) {
		Log(LogLevel::Error, "CheckInputType", "Invalid format type - only VIDEOINFOHEADER and VIDEOINFOHEADER2 are accepted");
		return E_INVALIDARG;
	}

    // We only accept NV12 or P010 input
    if (!IsEqualGUID(*mtIn->Type(), MEDIATYPE_Video) ||
		(!IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_NV12) && !IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_P010))) {
		Log(LogLevel::Error, "CheckInputType", "Invalid media subtype - only NV12 or P010 is accepted");
		return E_FAIL;
    }

    return NOERROR;
}

// Checks whether an input media type is compatible with an output media type
HRESULT CHopperRender::CheckTransform(const CMediaType* mtIn, const CMediaType* mtOut) {
	CheckPointer(mtIn, E_POINTER)
	CheckPointer(mtOut, E_POINTER)

	// We can transform NV12 or P010 input to P010 output
	if (!IsEqualGUID(*mtIn->Type(), MEDIATYPE_Video) ||
		(!IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_NV12) && !IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_P010)) || 
	    !IsEqualGUID(*mtOut->Subtype(), MEDIASUBTYPE_P010)) {
		Log(LogLevel::Error, "CheckTransform", "Incompatible media types for transformation");
		return E_FAIL;
    }

    return NOERROR;
}

// Sets the output pin's buffer requirements
HRESULT CHopperRender::DecideBufferSize(IMemAllocator* pAlloc, ALLOCATOR_PROPERTIES* ppropInputRequest) {
	CheckPointer(pAlloc, E_POINTER)
	CheckPointer(ppropInputRequest, E_POINTER)

    if (m_pInput->IsConnected() == FALSE) {
		Log(LogLevel::Error, "DecideBufferSize", "Input pin is not connected");
		return E_UNEXPECTED;
    }

    ppropInputRequest->cBuffers = 15;
    ppropInputRequest->cbBuffer = m_pOutput->CurrentMediaType().GetSampleSize();
    ppropInputRequest->cbAlign = 1;
    ppropInputRequest->cbPrefix = 0;

    ASSERT(ppropInputRequest->cbBuffer);

    // Ask the allocator to reserve us some sample memory, NOTE the function
    // can succeed (that is return NOERROR) but still not have allocated the
    // memory that we requested, so we must check we got whatever we wanted
    ALLOCATOR_PROPERTIES Actual;
    HRESULT hr = pAlloc->SetProperties(ppropInputRequest, &Actual);
    if (FAILED(hr)) {
		Log(LogLevel::Error, "DecideBufferSize", "Failed to set allocator properties");
		return hr;
    }

    if (ppropInputRequest->cBuffers > Actual.cBuffers ||
		ppropInputRequest->cbBuffer > Actual.cbBuffer) {
		Log(LogLevel::Error, "DecideBufferSize", "Allocator did not allocate requested buffer size");
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
    
    long biWidth, biHeight;
    unsigned int dwX, dwY;
    REFERENCE_TIME avgTimePerFrame;
    DWORD dwBitRate, dwBitErrorRate, dwCopyProtectFlags, dwInterlaceFlags, dwControlFlags, dwReserved1, dwReserved2;
    
    // Check if input is VIDEOINFOHEADER or VIDEOINFOHEADER2
    if (*mtIn->FormatType() == FORMAT_VideoInfo) {
        auto pvi = (VIDEOINFOHEADER*)mtIn->Format();
        biWidth = pvi->bmiHeader.biWidth;
        biHeight = abs(pvi->bmiHeader.biHeight);
        avgTimePerFrame = pvi->AvgTimePerFrame;
        dwBitRate = pvi->dwBitRate;
        dwBitErrorRate = pvi->dwBitErrorRate;
        dwX = biWidth;
        dwY = biHeight;
        dwCopyProtectFlags = 0;
        dwInterlaceFlags = 0;
        dwControlFlags = 0;
        dwReserved1 = 0;
        dwReserved2 = 0;
    } else {
        auto pvi2 = (VIDEOINFOHEADER2*)mtIn->Format();
        biWidth = pvi2->bmiHeader.biWidth;
        biHeight = abs(pvi2->bmiHeader.biHeight);
        avgTimePerFrame = pvi2->AvgTimePerFrame;
        dwBitRate = pvi2->dwBitRate;
        dwBitErrorRate = pvi2->dwBitErrorRate;
        dwX = pvi2->dwPictAspectRatioX;
        dwY = pvi2->dwPictAspectRatioY;
        dwCopyProtectFlags = pvi2->dwCopyProtectFlags;
        dwInterlaceFlags = pvi2->dwInterlaceFlags;
        dwControlFlags = pvi2->dwControlFlags;
        dwReserved1 = pvi2->dwReserved1;
        dwReserved2 = pvi2->dwReserved2;
    }

	// Retrieve the input frame time and dimensions
	if (avgTimePerFrame > 0)
		m_rtSourceFrameTime = avgTimePerFrame;
		m_rtCurrPlaybackFrameTime = m_rtSourceFrameTime;
	m_iDimX = biWidth;
	m_iDimY = biHeight;
    
    const GUID guid = MEDIASUBTYPE_P010;

    // Set the VideoInfoHeader2 information for the output media type
	VIDEOINFOHEADER2* vih2 = (VIDEOINFOHEADER2*)pMediaType->ReallocFormatBuffer(sizeof(VIDEOINFOHEADER2));
    memset(vih2, 0, sizeof(VIDEOINFOHEADER2));
    vih2->rcSource.right = vih2->rcTarget.right = biWidth;
    vih2->rcSource.bottom = vih2->rcTarget.bottom = biHeight;
    vih2->AvgTimePerFrame = avgTimePerFrame;
    vih2->dwPictAspectRatioX = dwX;
    vih2->dwPictAspectRatioY = dwY;
    vih2->dwBitRate = dwBitRate;
    vih2->dwBitErrorRate = dwBitErrorRate;
    vih2->dwCopyProtectFlags = dwCopyProtectFlags;
    vih2->dwInterlaceFlags = dwInterlaceFlags;
    vih2->dwControlFlags = dwControlFlags;
    vih2->dwReserved1 = dwReserved1;
    vih2->dwReserved2 = dwReserved2;

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
IBaseFilter* GetFilterFromPin(IPin* pPin) {
    CheckPointer(pPin, nullptr);

    PIN_INFO pi;
	if (pPin && SUCCEEDED(pPin->QueryPinInfo(&pi))) {
		return pi.pFilter;
    }

    return nullptr;
}

// Called when a new sample (source frame) arrives
HRESULT CHopperRender::Transform(IMediaSample* pIn, IMediaSample* pOut) {
	CheckPointer(pIn, E_POINTER)
	CheckPointer(pOut, E_POINTER)

	// Check for dynamic format changes
	AM_MEDIA_TYPE* pmt = nullptr;
	HRESULT hrMediaType = pIn->GetMediaType(&pmt);
	bool bFormatChanged = false;
	
	if (hrMediaType == S_OK && pmt != nullptr) {
		// Input sample has a new media type - format change!
		Log(LogLevel::Info, "Transform", "Dynamic format change detected on input sample");
		
		// Extract new dimensions
		long newWidth = 0, newHeight = 0;
		if (pmt->formattype == FORMAT_VideoInfo) {
			VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)pmt->pbFormat;
			newWidth = pvi->bmiHeader.biWidth;
			newHeight = abs(pvi->bmiHeader.biHeight);
		} else if (pmt->formattype == FORMAT_VideoInfo2) {
			VIDEOINFOHEADER2* pvi2 = (VIDEOINFOHEADER2*)pmt->pbFormat;
			newWidth = pvi2->bmiHeader.biWidth;
			newHeight = abs(pvi2->bmiHeader.biHeight);
		}
		
		if (newWidth != m_iDimX || newHeight != m_iDimY) {
			char logMsg[256];
			sprintf_s(logMsg, "Resolution change detected: %dx%d -> %dx%d", 
					m_iDimX, m_iDimY, newWidth, newHeight);
			Log(LogLevel::Info, "Transform", logMsg);
			
			m_iDimX = newWidth;
			m_iDimY = newHeight;
			bFormatChanged = true;
			
			// Reset optical flow calculator for new resolution
			if (m_pofcOpticalFlowCalc) {
				delete m_pofcOpticalFlowCalc;
				m_pofcOpticalFlowCalc = nullptr;
			}
		}
		
		DeleteMediaType(pmt);
	} else {
		// No format change in sample, but check if we need to initialize dimensions
		CMediaType* mtIn = &m_pInput->CurrentMediaType();
		long currentWidth = 0, currentHeight = 0;
		
		if (*mtIn->FormatType() == FORMAT_VideoInfo) {
			VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)mtIn->Format();
			currentWidth = pvi->bmiHeader.biWidth;
			currentHeight = abs(pvi->bmiHeader.biHeight);
		} else if (*mtIn->FormatType() == FORMAT_VideoInfo2) {
			VIDEOINFOHEADER2* pvi2 = (VIDEOINFOHEADER2*)mtIn->Format();
			currentWidth = pvi2->bmiHeader.biWidth;
			currentHeight = abs(pvi2->bmiHeader.biHeight);
		}
		
		if (currentWidth != m_iDimX || currentHeight != m_iDimY) {
			char logMsg[256];
			sprintf_s(logMsg, "Resolution initialized/changed: %dx%d -> %dx%d", 
					m_iDimX, m_iDimY, currentWidth, currentHeight);
			Log(LogLevel::Info, "Transform", logMsg);
			
			m_iDimX = currentWidth;
			m_iDimY = currentHeight;
			bFormatChanged = true;
			
			// Reset optical flow calculator for new resolution
			if (m_pofcOpticalFlowCalc) {
				delete m_pofcOpticalFlowCalc;
				m_pofcOpticalFlowCalc = nullptr;
			}
		}
	}

	// Update the output media type when format changes or on first frame
	if (bFormatChanged || m_iFrameCounter == 0) {
		CMediaType mtOut;
		mtOut.SetType(&MEDIATYPE_Video);
		mtOut.SetFormatType(&FORMAT_VideoInfo2);
		mtOut.SetSubtype(&MEDIASUBTYPE_P010);
		
		UpdateVideoInfoHeader(&mtOut);
		
		// Try to set the new format on the output pin
		HRESULT hrAccept = m_pOutput->GetConnected()->QueryAccept(&mtOut);
		if (hrAccept == S_OK) {
			Log(LogLevel::Info, "Transform", "Downstream filter accepted new format");
			m_pOutput->SetMediaType(&mtOut);
			
			// Set the media type on the output sample to signal format change
			AM_MEDIA_TYPE* pmtOut = CreateMediaType(&mtOut);
			if (pmtOut) {
				pOut->SetMediaType(pmtOut);
				DeleteMediaType(pmtOut);
			}
		} else {
			char logMsg[128];
			sprintf_s(logMsg, "Downstream filter rejected format change (HRESULT: 0x%08X)", hrAccept);
			Log(LogLevel::Error, "Transform", logMsg);
			
			// Try to reconnect
			IFilterGraph* pGraph = nullptr;
			if (SUCCEEDED(m_pGraph->QueryInterface(IID_IFilterGraph, (void**)&pGraph))) {
				HRESULT hrReconnect = pGraph->Reconnect(m_pOutput);
				if (SUCCEEDED(hrReconnect)) {
					Log(LogLevel::Info, "Transform", "Successfully reconnected output pin with new format");
				} else {
					sprintf_s(logMsg, "Failed to reconnect output pin (HRESULT: 0x%08X)", hrReconnect);
					Log(LogLevel::Error, "Transform", logMsg);
				}
				pGraph->Release();
			}
		}
    }

	// Update the display refresh rate every 5 seconds if the option is enabled
	if (m_bUseDisplayFPS && (m_iFrameCounter % (5 * 10000000 / m_rtTargetFrameTime) == 0)) {
		useDisplayRefreshRate();
	}

	m_iFrameCounter++;

    HRESULT hr = DeliverToRenderer(pIn, pOut);
    if (FAILED(hr)) {
		Log(LogLevel::Error, "Transform", "Failed to deliver frame to renderer");
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

	m_iPeakTotalFrameDelta = 0;
	m_frameDeltaHistory.clear();
}

// Called when a new segment is started (by seeking or changing the playback speed)
HRESULT CHopperRender::NewSegment(REFERENCE_TIME tStart, REFERENCE_TIME tStop, double dRate) {
    // Calculate the current source playback frame time
	m_rtCurrPlaybackFrameTime = static_cast<REFERENCE_TIME>(static_cast<double>(m_rtSourceFrameTime) * (1.0 / dRate));

    UpdateInterpolationStatus();

	m_iFrameCounter = 0;
	m_rtCurrStartTime = -1; // Tells the DeliverToRenderer function that we are at the start of a new segment

    return __super::NewSegment(tStart, tStop, dRate);
}

// Calculates the stride expected by the renderer
long CHopperRender::CalculateOutputStride(long bufferSize) {
	long outputStride = m_iDimX; // Default to frame width

	// Check if the downstream filter is MPC-VR
	IPin* pDownstreamPin = m_pOutput->GetConnected();
	if (pDownstreamPin) {
		IBaseFilter* pDownstreamFilter = GetFilterFromPin(pDownstreamPin);
		if (pDownstreamFilter) {
			CLSID filterCLSID;
			if (SUCCEEDED(pDownstreamFilter->GetClassID(&filterCLSID))) {
				if (filterCLSID != CLSID_MPC_VR) {
					if (m_iDimY > 0 && bufferSize > 0) {
						outputStride = bufferSize / (m_iDimY * 3);
					}
				}
			}
			pDownstreamFilter->Release();
		}
	}

	return outputStride;
}

// Delivers the new samples (interpolated frames) to the renderer
HRESULT CHopperRender::DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut) {
	CheckPointer(pIn, E_POINTER)
	CheckPointer(pOut, E_POINTER)

	// Get pointers to the sample buffer
	unsigned char* pInBuffer;
    HRESULT hr = pIn->GetPointer(&pInBuffer);
    if (FAILED(hr)) {
		Log(LogLevel::Error, "DeliverToRenderer", "Failed to get input buffer pointer");
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
		hr = sideDataIn->GetSideData(IID_MediaSideDataDOVIMetadata, &sideDataBytes1, &sideDataSize1);
		hr = sideDataIn->GetSideData(IID_MediaSideDataDOVIRPU, &sideDataBytes2, &sideDataSize2);
		hr = sideDataIn->GetSideData(IID_MediaSideDataControlFlags, &sideDataBytes3, &sideDataSize3);
		hr = sideDataIn->GetSideData(IID_MediaSideDataHDR, &sideDataBytes4, &sideDataSize4);
		hr = sideDataIn->GetSideData(IID_MediaSideDataHDR10Plus, &sideDataBytes5, &sideDataSize5);
		hr = sideDataIn->GetSideData(IID_MediaSideDataHDRContentLightLevel, &sideDataBytes6, &sideDataSize6);
		hr = sideDataIn->GetSideData(IID_MediaSideDataEIA608CC, &sideDataBytes7, &sideDataSize7);
		hr = sideDataIn->GetSideData(IID_MediaSideData3DOffset, &sideDataBytes8, &sideDataSize8);
    }

    // Get the size of the output sample
    const long lOutSize = pOut->GetActualDataLength();

    // Initialize the Optical Flow Calculator
    if (m_pofcOpticalFlowCalc == nullptr) {
		const long inputStride = m_iDimX;
		const long outputStride = CalculateOutputStride(lOutSize);
		
		int deltaScalar;
		int neighborScalar;
		float blackLevel;
		float whiteLevel;
		int customResScalar;
		loadSettings(&deltaScalar, &neighborScalar, &blackLevel, &whiteLevel, &customResScalar);
		char logMsg[256];
		sprintf_s(logMsg, "Initializing Optical Flow Calculator with %dx%d (input stride: %d, output stride: %d)", 
				m_iDimX, m_iDimY, inputStride, outputStride);
		Log(LogLevel::Info, "DeliverToRenderer", logMsg);
		if (m_pInput->CurrentMediaType().subtype == MEDIASUBTYPE_P010) {
			m_pofcOpticalFlowCalc = new OpticalFlowCalcHDR(m_iDimY, m_iDimX, inputStride, outputStride, deltaScalar, neighborScalar, blackLevel, whiteLevel, customResScalar);
			Log(LogLevel::Info, "DeliverToRenderer", "Using HDR Optical Flow Calculator");
		} else {
			m_pofcOpticalFlowCalc = new OpticalFlowCalcSDR(m_iDimY, m_iDimX, inputStride, outputStride, deltaScalar, neighborScalar, blackLevel, whiteLevel, customResScalar);
			Log(LogLevel::Info, "DeliverToRenderer", "Using SDR Optical Flow Calculator");
		}
    }

    // Get the presentation times for the new output sample
    REFERENCE_TIME rtStartTime, rtEndTime;
    hr = pIn->GetTime(&rtStartTime, &rtEndTime);
    if (FAILED(hr)) {
		// Capture cards often don't provide valid timestamps, continue with default values
		Log(LogLevel::Info, "DeliverToRenderer", "Failed to get sample time, using defaults");
		rtStartTime = 0;
		rtEndTime = 0;
		m_bValidFrameTimes = false;
    } else {
		m_bValidFrameTimes = true;
    }

	// Reset our frame time if necessary and calculate the current number of intermediate frames needed
    if (m_rtCurrStartTime == -1) {
		// We are at the start of a new segment
		m_rtCurrStartTime = rtStartTime;
    }

	// Calculate the number of interpolated frames
	if (m_iIntActiveState == Active) {
		m_iNumIntFrames = (int)max(ceil((1.0 - m_dBlendingScalar) / ((double)m_rtTargetFrameTime / (double)m_rtCurrPlaybackFrameTime)), 1.0);
	} else {
		m_iNumIntFrames = 1;
	}

	if (m_iFrameCounter == 1) {
		char dbgLogMsg[256];
		sprintf_s(dbgLogMsg, "Adding %d additional frames for LIVE content", m_iBufferFrames);
		Log(LogLevel::Info, "DeliverToRenderer", dbgLogMsg);
		m_iNumIntFrames += m_iBufferFrames;
	}

    // Adjust the settings to process everything fast enough
    autoAdjustSettings();

    m_pofcOpticalFlowCalc->updateFrame(pInBuffer);

    if (m_iIntActiveState == Active && m_iFrameCounter >= 2) {
		// Calculate the optical flow (frame 1 to frame 2)
		m_pofcOpticalFlowCalc->calculateOpticalFlow();

		// Track peak total frame delta using sliding window (3 seconds)
		int framesIn3Seconds = (int)(3.0 * 10000000.0 / m_rtSourceFrameTime);
		
		// Add current frame delta to history
		FrameDeltaEntry entry;
		entry.frameNumber = m_iFrameCounter;
		entry.totalDelta = m_pofcOpticalFlowCalc->m_totalFrameDelta;
		m_frameDeltaHistory.push_back(entry);
		
		// Remove entries older than 3 seconds
		while (!m_frameDeltaHistory.empty() && 
		       (m_iFrameCounter - m_frameDeltaHistory.front().frameNumber) > framesIn3Seconds) {
			m_frameDeltaHistory.pop_front();
		}
		
		// Calculate peak from remaining entries
		m_iPeakTotalFrameDelta = 0;
		for (const auto& histEntry : m_frameDeltaHistory) {
			if (histEntry.totalDelta > m_iPeakTotalFrameDelta) {
				m_iPeakTotalFrameDelta = histEntry.totalDelta;
			}
		}
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
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to get delivery buffer");
				return hr;
			}
			// Use the input sample for the last output sample
		} else {
			pOutNew = pOut;
		}

		// Set the side data
		IMediaSideData* sideDataOut;
		if (SUCCEEDED(hr = pOutNew->QueryInterface(&sideDataOut))) {
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
			Log(LogLevel::Error, "DeliverToRenderer", "Failed to get output buffer pointer");
			return hr;
		}

		// Set the new start and end times
		if (m_iIntActiveState == Active) {
			rtStartTime = m_rtCurrStartTime;
			rtEndTime = rtStartTime + m_rtTargetFrameTime;
		}

		// Set the new times for the output sample
		if (m_bValidFrameTimes) {
			hr = pOutNew->SetTime(&rtStartTime, &rtEndTime);
			if (FAILED(hr)) {
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to set sample time");
				return hr;
			}
		}

		// Increment the frame time for the next sample
		m_rtCurrStartTime += m_rtTargetFrameTime;

		// Copy the media times
		LONGLONG llMediaStart, llMediaEnd;
		if (NOERROR == pIn->GetMediaTime(&llMediaStart, &llMediaEnd)) {
			hr = pOutNew->SetMediaTime(&llMediaStart, &llMediaEnd);
			if (FAILED(hr)) {
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to set media time");
				return hr;
			}
		}

		// Copy the Sync point property
		hr = pIn->IsSyncPoint();
		if (hr == S_OK) {
			hr = pOutNew->SetSyncPoint(TRUE);
			if (FAILED(hr)) {
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to set sync point");
				return hr;
			}
		} else if (hr == S_FALSE) {
			hr = pOutNew->SetSyncPoint(FALSE);
			if (FAILED(hr)) {
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to clear sync point");
				return hr;
			}
		} else {
			Log(LogLevel::Error, "DeliverToRenderer", "Unexpected result from IsSyncPoint");
			return E_UNEXPECTED;
		}

		// Copy the media type
		AM_MEDIA_TYPE* pMediaType;
		hr = pOut->GetMediaType(&pMediaType);
		if (FAILED(hr)) {
			Log(LogLevel::Error, "DeliverToRenderer", "Failed to get media type");
			return hr;
		}
		hr = pOutNew->SetMediaType(pMediaType);
		if (FAILED(hr)) {
			Log(LogLevel::Error, "DeliverToRenderer", "Failed to set media type");
			return hr;
		}
		DeleteMediaType(pMediaType);

		// Copy the preroll property
		hr = pIn->IsPreroll();
		if (hr == S_OK) {
			hr = pOutNew->SetPreroll(TRUE);
			if (FAILED(hr)) {
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to set preroll");
				return hr;
			}
		} else if (hr == S_FALSE) {
			hr = pOutNew->SetPreroll(FALSE);
			if (FAILED(hr)) {
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to clear preroll");
				return hr;
			}
		} else {
			Log(LogLevel::Error, "DeliverToRenderer", "Unexpected result from IsPreroll");
			return E_UNEXPECTED;
		}

		// Copy the discontinuity property
		hr = pIn->IsDiscontinuity();
		if (hr == S_OK) {
			hr = pOutNew->SetDiscontinuity(TRUE);
			if (FAILED(hr)) {
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to set discontinuity");
				return hr;
			}
		} else if (hr == S_FALSE) {
			hr = pOutNew->SetDiscontinuity(FALSE);
			if (FAILED(hr)) {
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to clear discontinuity");
				return hr;
			}
		} else {
			Log(LogLevel::Error, "DeliverToRenderer", "Unexpected result from IsDiscontinuity");
			return E_UNEXPECTED;
		}

		// Copy the actual data length
		hr = pOutNew->SetActualDataLength(lOutSize);
		if (FAILED(hr)) {
			Log(LogLevel::Error, "DeliverToRenderer", "Failed to set actual data length");
			return hr;
		}

		// Interpolate the frame if necessary
		if (m_iIntActiveState == Active && m_iFrameCounter >= 2 && m_pofcOpticalFlowCalc->m_totalFrameDelta < m_iSceneChangeThreshold) {
			m_pofcOpticalFlowCalc->warpFrames(m_dBlendingScalar, m_iFrameOutput);
		} else {
			m_pofcOpticalFlowCalc->copyFrame();
		}

		// Download the result to the output buffer
		m_pofcOpticalFlowCalc->downloadFrame(pOutNewBuffer);

		// Retrieve how long the warp calculation took
		m_dTotalWarpDuration += m_pofcOpticalFlowCalc->m_warpCalcTime;

		// Increase the blending scalar
		if (m_iIntActiveState == Active) {
			m_dBlendingScalar += (double)m_rtTargetFrameTime / (double)m_rtCurrPlaybackFrameTime;
			if (m_dBlendingScalar >= 1.0) {
				m_dBlendingScalar -= 1.0;
			}
		}

		// Deliver the new output sample downstream
		// We don't need to deliver the last sample, as it is automatically delivered by the caller
		if (iIntFrameNum < (m_iNumIntFrames - 1)) {
			hr = m_pOutput->Deliver(pOutNew);
			if (FAILED(hr)) {
				Log(LogLevel::Error, "DeliverToRenderer", "Failed to deliver output sample");
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
		Log(LogLevel::Error, "GetPages", "Failed to allocate memory for property pages");
		return E_OUTOFMEMORY;
    }

    *(pPages->pElems) = CLSID_HopperRenderSettings;
    return NOERROR;
}

// Return the current settings selected
STDMETHODIMP CHopperRender::GetCurrentSettings(bool* pbActivated,
											   int* piFrameOutput,
											   double* pdTargetFPS,
											   bool* pbUseDsiplayFPS,
											   int* piDeltaScalar,
											   int* piNeighborScalar,
											   int* piBlackLevel,
											   int* piWhiteLevel,
											   int* piSceneChangeThreshold,
											   int* piIntActiveState,
											   double* pdSourceFPS,
											   double* pdOFCCalcTime,
											   double* pdAVGOFCCalcTime,
											   double* pdPeakOFCCalcTime,
											   double* pdWarpCalcTime,
											   int* piDimX,
											   int* piDimY,
											   int* piLowDimX,
											   int* piLowDimY,
											   unsigned int* piTotalFrameDelta,
											   unsigned int* piBufferFrames) {
    CAutoLock cAutolock(&m_csHopperRenderLock);
	CheckPointer(pbActivated, E_POINTER)
	CheckPointer(piFrameOutput, E_POINTER)
	CheckPointer(pdTargetFPS, E_POINTER)
	CheckPointer(pbUseDsiplayFPS, E_POINTER)
	CheckPointer(piDeltaScalar, E_POINTER)
	CheckPointer(piNeighborScalar, E_POINTER)
	CheckPointer(piBlackLevel, E_POINTER)
	CheckPointer(piWhiteLevel, E_POINTER)
	CheckPointer(piSceneChangeThreshold, E_POINTER)
	CheckPointer(piIntActiveState, E_POINTER)
	CheckPointer(pdSourceFPS, E_POINTER)
	CheckPointer(pdOFCCalcTime, E_POINTER)
	CheckPointer(pdAVGOFCCalcTime, E_POINTER)
	CheckPointer(pdPeakOFCCalcTime, E_POINTER)
	CheckPointer(pdWarpCalcTime, E_POINTER)
	CheckPointer(piDimX, E_POINTER)
	CheckPointer(piDimY, E_POINTER)
	CheckPointer(piLowDimX, E_POINTER)
	CheckPointer(piLowDimY, E_POINTER)
	CheckPointer(piTotalFrameDelta, E_POINTER)
	CheckPointer(piBufferFrames, E_POINTER)

	// Optical Flow Calculator not initialized yet
	if (m_pofcOpticalFlowCalc == nullptr) {
		int deltaScalar;
		int neighborScalar;
		float blackLevel;
		float whiteLevel;
		int customResScalar;
	    loadSettings(&deltaScalar, &neighborScalar, &blackLevel, &whiteLevel, &customResScalar);
		*pbActivated = m_iIntActiveState != 0;
		*piFrameOutput = m_iFrameOutput;
		if (*pdTargetFPS == 0.0 || m_bUseDisplayFPS) {
			*pdTargetFPS = 10000000.0 / static_cast<double>(m_rtTargetFrameTime);
		}
		*pbUseDsiplayFPS = m_bUseDisplayFPS;
		*piDeltaScalar = deltaScalar;
		*piNeighborScalar = neighborScalar;
		*piBlackLevel = blackLevel / 256.0f;
		*piWhiteLevel = whiteLevel / 256.0f;
		*piSceneChangeThreshold = m_iSceneChangeThreshold;
		*piIntActiveState = Active;
		*pdSourceFPS = 0.0;
		*pdOFCCalcTime = 0.0;
		*pdAVGOFCCalcTime = 0.0;
		*pdPeakOFCCalcTime = 0.0;
		*pdWarpCalcTime = 0.0;
		*piDimX = 0;
		*piDimY = 0;
		*piLowDimX = 0;
		*piLowDimY = 0;
		*piTotalFrameDelta = 0;
		*piBufferFrames = m_iBufferFrames;
	} else {
		*pbActivated = m_iIntActiveState != 0;
		*piFrameOutput = m_iFrameOutput;
		if (*pdTargetFPS == 0.0 || m_bUseDisplayFPS) {
			*pdTargetFPS = 10000000.0 / static_cast<double>(m_rtTargetFrameTime);
		}
		*pbUseDsiplayFPS = m_bUseDisplayFPS;
		*piDeltaScalar = m_pofcOpticalFlowCalc->m_deltaScalar;
		*piNeighborScalar = m_pofcOpticalFlowCalc->m_neighborBiasScalar;
		*piBlackLevel = (int)(m_pofcOpticalFlowCalc->m_outputBlackLevel) >> 8;
		*piWhiteLevel = (int)(m_pofcOpticalFlowCalc->m_outputWhiteLevel) >> 8;
		*piSceneChangeThreshold = m_iSceneChangeThreshold;
		*piIntActiveState = m_iIntActiveState;
		*pdSourceFPS = 10000000.0 / static_cast<double>(m_rtCurrPlaybackFrameTime);
		*pdOFCCalcTime = 1000.0 * m_pofcOpticalFlowCalc->m_ofcCalcTime;
		*pdAVGOFCCalcTime = 1000.0 * m_pofcOpticalFlowCalc->m_ofcAvgCalcTime;
		*pdPeakOFCCalcTime = 1000.0 * m_pofcOpticalFlowCalc->m_ofcPeakCalcTime;
		*pdWarpCalcTime = 1000.0 * m_dTotalWarpDuration;
		*piDimX = m_iDimX;
		*piDimY = m_iDimY;
		*piLowDimX = m_pofcOpticalFlowCalc->m_opticalFlowFrameWidth;
		*piLowDimY = m_pofcOpticalFlowCalc->m_opticalFlowFrameHeight;
		*piTotalFrameDelta = m_iPeakTotalFrameDelta;
		*piBufferFrames = m_iBufferFrames;
	}
	return NOERROR;
}

// Apply the new settings
STDMETHODIMP CHopperRender::UpdateUserSettings(bool bActivated,
											   int iFrameOutput,
											   double dTargetFPS,
											   bool bUseDisplayFPS,
											   int iDeltaScalar,
											   int iNeighborScalar,
											   int iBlackLevel,
											   int iWhiteLevel,
											   int iSceneChangeThreshold,
											   unsigned int iBufferFrames) {
	CAutoLock cAutolock(&m_csHopperRenderLock);

	if (!bActivated) {
		m_iIntActiveState = Deactivated;
	} else if (!m_iIntActiveState) {
		m_iIntActiveState = Active;
	}
	m_iFrameOutput = static_cast<FrameOutput>(iFrameOutput);
	if (dTargetFPS > 0.0 && !bUseDisplayFPS) {
	    m_rtTargetFrameTime = (1.0 / (double)dTargetFPS) * 1e7;
	} else {
	    useDisplayRefreshRate();
	}
	m_bUseDisplayFPS = bUseDisplayFPS;
	m_iSceneChangeThreshold = iSceneChangeThreshold;
	m_iBufferFrames = iBufferFrames;
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
    LONG result = 1;
    LONG result2 = 1;

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

		valueName = L"Use Display FPS";
		result2 = RegQueryValueEx(hKey, valueName, NULL, NULL,
					reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value2 > 0 && (!value || result2 != 0)) {
			m_rtTargetFrameTime = (1.0 / (double)value2) * 1e7;
			m_bUseDisplayFPS = false;
		} else {
			useDisplayRefreshRate();
			m_bUseDisplayFPS = true;
		}

		// Load the delta scalar
		valueName = L"DeltaScalar";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0 && value <= 10) {
			*deltaScalar = value;
		} else {
			*deltaScalar = DEFAULT_DELTA_SCALAR;
		}

		// Load the neighbor scalar
		valueName = L"NeighborScalar";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0 && value <= 10) {
			*neighborScalar = value;
		} else {
			*neighborScalar = DEFAULT_NEIGHBOR_SCALAR;
		}

		// Load the black level
		valueName = L"BlackLevel";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0 && value <= 255) {
			*blackLevel = (float)(value << 8);
		} else {
			*blackLevel = (float)DEFAULT_BLACK_LEVEL;
		}

		// Load the white level
		valueName = L"WhiteLevel";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0 && value <= 255) {
			*whiteLevel = (float)(value << 8);
		} else {
			*whiteLevel = (float)(DEFAULT_WHITE_LEVEL << 8);
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

		// Load the scene change threshold
		valueName = L"SceneChangeThreshold";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0) {
			m_iSceneChangeThreshold = value;
		} else {
			m_iSceneChangeThreshold = DEFAULT_SCENE_CHANGE_THRESHOLD;
		}

		// Load the buffer frames
		valueName = L"BufferFrames";
		result = RegQueryValueEx(hKey, valueName, NULL, NULL,
					 reinterpret_cast<BYTE*>(&value), &dataSize);
		if (result == 0 && value >= 0 && value <= 1000) {
			m_iBufferFrames = value;
		} else {
			m_iBufferFrames = DEFAULT_BUFFER_FRAMES;
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
