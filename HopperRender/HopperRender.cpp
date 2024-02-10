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

#include <numeric>

#include "resource.h"


// Debug message function
void DebugMessage(const std::string& message) {
	#if defined(DEBUG)
    const std::string m_debugMessage = message + "\n";
    OutputDebugStringA(m_debugMessage.c_str());
	#endif
}


// Input/Output pin types
constexpr AMOVIESETUP_MEDIATYPE sudPinTypes =
{
    &MEDIATYPE_Video,       // Major type
    &MEDIASUBTYPE_RGB24     // Minor type
};


// Input/Output pin information
const AMOVIESETUP_PIN sudpPins[] =
{
    { L"Input",          // Pins string name
      FALSE,                // Is it rendered
      FALSE,                // Is it an output
      FALSE,                // Are we allowed none
      FALSE,                // And allowed many
      &CLSID_NULL,          // Connects to filter
      nullptr,              // Connects to pin
      1,                    // Number of types
      &sudPinTypes          // Pin information
    },
    { L"Output",         // Pins string name
      FALSE,                // Is it rendered
      TRUE,                 // Is it an output
      FALSE,                // Are we allowed none
      FALSE,                // And allowed many
      &CLSID_NULL,          // Connects to filter
      nullptr,              // Connects to pin
      1,                    // Number of types
      &sudPinTypes          // Pin information
    }
};


// Filter information
constexpr AMOVIESETUP_FILTER sudHopperRender =
{
    &CLSID_HopperRender,    // Filter CLSID
    L"HopperRender",        // String name
    MERIT_DO_NOT_USE,       // Filter merit
    2,                      // Number of pins
    sudpPins                // Pin information
};


// List of class IDs and creator functions for the class factory. This
// provides the link between the OLE entry point in the DLL and an object
// being created. The class factory will call the static CreateInstance
CFactoryTemplate g_Templates[] = {
    { L"HopperRender"
    , &CLSID_HopperRender
    , CHopperRender::CreateInstance
    , nullptr
    , &sudHopperRender }
  ,
    { L"HopperRender Settings"
    , &CLSID_HopperRenderSettings
    , CHopperRenderSettings::CreateInstance }
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
    DWORD  dwReason,
    LPVOID lpReserved) {

    return DllEntryPoint((HINSTANCE)(hModule), dwReason, lpReserved);
}


// Constructor
CHopperRender::CHopperRender(TCHAR* tszName,
    LPUNKNOWN punk,
    HRESULT* phr) :
    CTransformFilter(tszName, punk, CLSID_HopperRender),
    m_iEffect(IDC_NONE),
    m_lBufferRequest(12),
    CPersistStream(punk, phr),
    m_bBisNewest(true) {

    char sz[60];

    GetProfileStringA("Quartz", "EffectStart", "40", sz, 60);
    m_iNumSteps = atoi(sz);

    GetProfileStringA("Quartz", "EffectLength", "192", sz, 60);
    m_iMaxOffsetDivider = atoi(sz);

    // Initialize the past frame durations
    std::fill(std::begin(m_rtPastFrameDurations), std::end(m_rtPastFrameDurations), 0);
}


// Provide the way for COM to create a HopperRender object
CUnknown* CHopperRender::CreateInstance(LPUNKNOWN punk, HRESULT* phr) {
    ASSERT(phr);

    CHopperRender* pNewObject = new CHopperRender(NAME("HopperRender"), punk, phr);

    if (pNewObject == nullptr) {
        if (phr)
            *phr = E_OUTOFMEMORY;
    }
    return pNewObject;
}


// Reveals IIPEffect and ISpecifyPropertyPages
STDMETHODIMP CHopperRender::NonDelegatingQueryInterface(REFIID riid, void** ppv) {
    CheckPointer(ppv, E_POINTER)

    if (riid == IID_IIPEffect) {
        return GetInterface((IIPEffect*)this, ppv);
    } else if (riid == IID_ISpecifyPropertyPages) {
        return GetInterface((ISpecifyPropertyPages*)this, ppv);
    } else {
        return CTransformFilter::NonDelegatingQueryInterface(riid, ppv);
    }
}


// Checks whether a specified media type is acceptable for input.
HRESULT CHopperRender::CheckInputType(const CMediaType* mtIn) {
    CheckPointer(mtIn, E_POINTER)

    // Check this is a VIDEOINFOHEADER type
    if (*mtIn->FormatType() != FORMAT_VideoInfo) {
        return E_INVALIDARG;
    }

    // Can we transform this type
    if (IsEqualGUID(*mtIn->Type(), MEDIATYPE_Video) && IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_RGB24)) {
        VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)mtIn->Format();
        if (pvi->bmiHeader.biBitCount == 24) {
        	return NOERROR;
		}
    }
    return E_FAIL;
}


// Checks whether an input media type is compatible with an output media type.
HRESULT CHopperRender::CheckTransform(const CMediaType* mtIn, const CMediaType* mtOut) {
    CheckPointer(mtIn, E_POINTER)
    CheckPointer(mtOut, E_POINTER)

    if (IsEqualGUID(*mtIn->Type(), MEDIATYPE_Video) && IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_RGB24)) {
        VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)mtIn->Format();
        if (pvi->bmiHeader.biBitCount == 24) {
			if (*mtIn == *mtOut) {
	            return NOERROR;
	        }
        }
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

    ppropInputRequest->cBuffers = 1;
    ppropInputRequest->cbBuffer = m_pInput->CurrentMediaType().GetSampleSize() * m_lBufferRequest;
    ASSERT(ppropInputRequest->cbBuffer);

    // Ask the allocator to reserve us some sample memory, NOTE the function
    // can succeed (that is return NOERROR) but still not have allocated the
    // memory that we requested, so we must check we got whatever we wanted
    ALLOCATOR_PROPERTIES Actual;
    hr = pAlloc->SetProperties(ppropInputRequest, &Actual);
    if (FAILED(hr)) {
        return hr;
    }

    ASSERT(Actual.cBuffers == 1);

    if (ppropInputRequest->cBuffers > Actual.cBuffers ||
        ppropInputRequest->cbBuffer > Actual.cbBuffer) {
        return E_FAIL;
    }
    return NOERROR;
}


// Retrieves a preferred media type for the output pin
HRESULT CHopperRender::GetMediaType(int iPosition, CMediaType* pMediaType) {
    // This should never happen
    if (iPosition < 0) {
        return E_INVALIDARG;
    }

    // Do we have more items to offer
    if (iPosition > 0) {
        return VFW_S_NO_MORE_ITEMS;
    }

    CheckPointer(pMediaType, E_POINTER)
    *pMediaType = m_pInput->CurrentMediaType();

    return NOERROR;
}


// Transforms an input sample to produce an output sample
HRESULT CHopperRender::Transform(IMediaSample* pIn, IMediaSample* pOut) {
    CheckPointer(pIn, E_POINTER)
	CheckPointer(pOut, E_POINTER)

	// Increment the frame counter
	m_iFrameCounter++;

    // Copy the properties across
    HRESULT hr = DeliverToRenderer(pIn, pOut, 166667);
    if (FAILED(hr)) {
        return hr;
    }

    return NOERROR;
}


// Delivers the first sample to the renderer
HRESULT CHopperRender::DeliverToRenderer(IMediaSample* pIn, IMediaSample* pOut, REFERENCE_TIME rtAvgFrameTimeTarget) {
    CheckPointer(pIn, E_POINTER)
    CheckPointer(pOut, E_POINTER)

    // Get pointers to the sample buffer
    BYTE* pInBuffer;
    HRESULT hr = pIn->GetPointer(&pInBuffer);
    if (FAILED(hr)) {
        return hr;
    }

    // Get the size of the samples
    const long lInSize = pIn->GetActualDataLength();

    // Assemble the output samples
    IMediaSample* pOutNew;
    BYTE* pOutNewBuffer;
    int iNumSamples = 1;

    for (int iIntFrameNum = 0; iIntFrameNum < iNumSamples; ++iIntFrameNum) {
        // Use the input sample for the first output sample
        if (iIntFrameNum == 0) {
            pOutNew = pOut;
        // Create a new output sample
        } else {
            pOutNew = nullptr;
            hr = m_pOutput->GetDeliveryBuffer(&pOutNew, nullptr, nullptr, 0);
            if (FAILED(hr)) {
                return hr;
            }
        }

        // Get the buffer pointer for the new output sample
        hr = pOutNew->GetPointer(&pOutNewBuffer);
        if (FAILED(hr)) {
            return hr;
        }

        // Set the presentation time for the new output sample
        REFERENCE_TIME rtStartTime, rtEndTime;
        hr = pIn->GetTime(&rtStartTime, &rtEndTime);
        if (FAILED(hr)) {
            return hr;
        }

        // Update the past frame durations
        if (iIntFrameNum == 0) {
            if (rtStartTime - m_rtLastStartTime > 0) {
                m_rtPastFrameDurations[m_iFrameCounter % 10] = rtStartTime - m_rtLastStartTime;
            }
		}

        // Check if interpolation is necessary
        if (m_iFrameCounter % 10 == 0) {
            const REFERENCE_TIME rtSum = std::accumulate(std::begin(m_rtPastFrameDurations), std::end(m_rtPastFrameDurations), static_cast<REFERENCE_TIME>(0));
            m_rtAvgSourceFrameTime = rtSum / 10;

            if (m_rtAvgSourceFrameTime > (rtAvgFrameTimeTarget + 833)) {
                m_bIntNeeded = true;
            } else {
                m_bIntNeeded = false;
            }
        }

        // Update the stats
        //if (iIntFrameNum == 0) {
            //m_IntActive = m_bIntNeeded;
            //m_SourceFPS = 10000000.0 / static_cast<double>(m_rtAvgSourceFrameTime);
            //ScribbleToStream(nullptr);
        //}

        if (m_bIntNeeded) { // Interpolation needed

            if (iIntFrameNum == 0) {
                // Print the original start and end times
                DebugMessage("ACINT | Start time: " + std::to_string(rtStartTime) + " End time: " + std::to_string(rtEndTime) + " Delta: " + std::to_string(rtEndTime - rtStartTime) + " Start2-Start1: " + std::to_string(rtStartTime - m_rtLastStartTime) + " AVG S2-S1: " + std::to_string(m_rtAvgSourceFrameTime));

                // Reset our frame time if necessary and calculate the current number of intermediate frames needed
                if (rtStartTime < m_rtLastStartTime) {
                    m_rtCurrStartTime = rtStartTime;
                    iNumSamples = static_cast<int>(ceil((static_cast<double>(rtEndTime) - static_cast<double>(rtStartTime)) / static_cast<double>(rtAvgFrameTimeTarget)));
                } else {
                    iNumSamples = static_cast<int>(ceil(static_cast<double>((m_rtAvgSourceFrameTime + rtStartTime - m_rtCurrStartTime)) / static_cast<double>(rtAvgFrameTimeTarget)));
                }
                m_rtLastStartTime = rtStartTime;
            }

            // Ensure there was no mistake
            if (iNumSamples < 1) {
            	iNumSamples = 1;
			}

            // Set the new start and end times
            rtStartTime = m_rtCurrStartTime;
            rtEndTime = rtStartTime + rtAvgFrameTimeTarget;

        } else { // No interpolation needed
            DebugMessage("NOINT | Start time: " + std::to_string(rtStartTime) + " End time: " + std::to_string(rtEndTime) + " Delta: " + std::to_string(rtEndTime - rtStartTime) + " Start2-Start1: " + std::to_string(rtStartTime - m_rtLastStartTime) + " AVG S2-S1: " + std::to_string(m_rtAvgSourceFrameTime));
            iNumSamples = 1;
            m_rtCurrStartTime = rtStartTime;
            m_rtLastStartTime = rtStartTime;
        }

        // Set the new time for the output sample
        hr = pOutNew->SetTime(&rtStartTime, &rtEndTime);
        if (FAILED(hr)) {
            return hr;
        }

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
        } else {  // an unexpected error has occured...
            return E_UNEXPECTED;
        }

        // Copy the media type
        AM_MEDIA_TYPE* pMediaType;
        hr = pIn->GetMediaType(&pMediaType);
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
        } else {  // an unexpected error has occured...
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
        } else {  // an unexpected error has occured...
            return E_UNEXPECTED;
        }

        // Copy the actual data length
        hr = pOutNew->SetActualDataLength(lInSize);
        if (FAILED(hr)) {
            return hr;
        }

        // Interpolate the frame
        hr = InterpolateFrame(pInBuffer, pOutNewBuffer, iIntFrameNum, iNumSamples);
        if (FAILED(hr)) {
            return hr;
        }

        // Deliver the new output sample downstream
        if (iIntFrameNum > 0) {
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


HRESULT CHopperRender::InterpolateFrame(BYTE* pInBuffer, BYTE* pOutBuffer, int iIntFrameNum, int iNumSamples) {
    // Get the image properties from the BITMAPINFOHEADER
    AM_MEDIA_TYPE* pType = &m_pInput->CurrentMediaType();
    VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)pType->pbFormat;
    ASSERT(pvi);
    unsigned int dimX = pvi->bmiHeader.biWidth;
    unsigned int dimY = pvi->bmiHeader.biHeight;

    // Initialize the GPU Arrays if they haven't been initialized yet
    if (!m_gpuFrameA.isInitialized()) {
        m_gpuFrameA.init({ 3, dimY, dimX });
        m_gpuFrameB.init({ 3, dimY, dimX });
        m_gpuFrameB.fillData(pInBuffer);
        //memcpy(pOutBuffer, pInBuffer, 3 * dimY * dimX);
        m_ofcOpticalFlowCalc.init(dimY, dimX);
    }

    // Either fill the A or B frame with the new data, so that
    // we always have the current frame and the previous frame
    if (iIntFrameNum == 0) {
        if (m_bBisNewest) {
            m_gpuFrameA.fillData(pInBuffer);
        } else {
            m_gpuFrameB.fillData(pInBuffer);
        }
    }

    // Blend the frames together
    if (m_bBisNewest) {
        m_ofcOpticalFlowCalc.blendFrames(m_gpuFrameA, m_gpuFrameB, iIntFrameNum, iNumSamples);
    } else {
        m_ofcOpticalFlowCalc.blendFrames(m_gpuFrameB, m_gpuFrameA, iIntFrameNum, iNumSamples);
    }
    m_ofcOpticalFlowCalc.warpedFrame.download(pOutBuffer);

    // Update the frame order
    if (iIntFrameNum == 0) {
    	m_bBisNewest = !m_bBisNewest;
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

    WRITEOUT(m_iEffect)
    WRITEOUT(m_iNumSteps)
    WRITEOUT(m_iMaxOffsetDivider)

    return NOERROR;
}


// Likewise overriden to restore our state from a stream
HRESULT CHopperRender::ReadFromStream(IStream* pStream) {
    HRESULT hr;

    READIN(m_iEffect)
    READIN(m_iNumSteps)
    READIN(m_iMaxOffsetDivider)

    return NOERROR;
}


// Returns the clsid's of the property pages we support
STDMETHODIMP CHopperRender::GetPages(CAUUID* pPages) {
    CheckPointer(pPages, E_POINTER)

    pPages->cElems = 1;
    pPages->pElems = (GUID*)CoTaskMemAlloc(sizeof(GUID));
    if (pPages->pElems == nullptr) {
        return E_OUTOFMEMORY;
    }

    *(pPages->pElems) = CLSID_HopperRenderSettings;
    return NOERROR;
}


// Return the current effect selected
STDMETHODIMP CHopperRender::get_IPEffect(int* IPEffect, int* pNumSteps, int* pMaxOffsetDivider) {
    CAutoLock cAutolock(&m_csHopperRenderLock);
    CheckPointer(IPEffect, E_POINTER)
    CheckPointer(pNumSteps, E_POINTER)
    CheckPointer(pMaxOffsetDivider, E_POINTER)

    *IPEffect = m_iEffect;
    *pNumSteps = m_iNumSteps;
    *pMaxOffsetDivider = m_iMaxOffsetDivider;

    return NOERROR;
}


// Set the required video effect
STDMETHODIMP CHopperRender::put_IPEffect(int IPEffect, int numSteps, int maxOffsetDivider) {
    CAutoLock cAutolock(&m_csHopperRenderLock);

    m_iEffect = IPEffect;
    m_iNumSteps = numSteps;
    m_iMaxOffsetDivider = maxOffsetDivider;

    SetDirty(TRUE);
    return NOERROR;
}