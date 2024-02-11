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
	CPersistStream(punk, phr),
	m_bActivated(IDC_ON),
    m_iNumSteps(40),
    m_iMaxOffsetDivider(192),
    m_lBufferRequest(12),
    m_bBisNewest(true),
    m_iFrameCounter(0),
    m_rtCurrStartTime(LONGLONG_MAX),
    m_rtLastStartTime(LONGLONG_MAX),
    m_bIntNeeded(false),
    m_rtAvgSourceFrameTime(0) {

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

    if (riid == IID_SettingsInterface) {
        return GetInterface((SettingsInterface*)this, ppv);
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

            if (m_bActivated) {
                if (m_rtAvgSourceFrameTime > (rtAvgFrameTimeTarget + 833)) {
                    m_bIntNeeded = true;
                } else {
                    m_bIntNeeded = false;
                }
            } else {
				m_bIntNeeded = false;
			}
            
        }

        if (m_bIntNeeded) { // Interpolation needed

            if (iIntFrameNum == 0) {
                // Print the original start and end times
                //DebugMessage("ACINT | Start time: " + std::to_string(rtStartTime) + " End time: " + std::to_string(rtEndTime) + " Delta: " + std::to_string(rtEndTime - rtStartTime) + " Start2-Start1: " + std::to_string(rtStartTime - m_rtLastStartTime) + " AVG S2-S1: " + std::to_string(m_rtAvgSourceFrameTime));

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
            //DebugMessage("NOINT | Start time: " + std::to_string(rtStartTime) + " End time: " + std::to_string(rtEndTime) + " Delta: " + std::to_string(rtEndTime - rtStartTime) + " Start2-Start1: " + std::to_string(rtStartTime - m_rtLastStartTime) + " AVG S2-S1: " + std::to_string(m_rtAvgSourceFrameTime));
            iNumSamples = 1;
            m_rtCurrStartTime = rtStartTime;
            m_rtLastStartTime = rtStartTime;
        }

        // Set the new time for the output sample
        hr = pOutNew->SetTime(&rtStartTime, &rtEndTime);
        if (FAILED(hr)) {
            return hr;
        }

        // Calculate the scalar for the interpolation
        //const double dScalar = max(min((static_cast<double>(m_rtCurrStartTime) - static_cast<double>(m_rtLastStartTime)) / static_cast<double>(m_rtAvgSourceFrameTime), 1.0), 0.0);
        const double dScalar = static_cast<double>(iIntFrameNum) / static_cast<double>(iNumSamples);

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
        hr = InterpolateFrame(pInBuffer, pOutNewBuffer, dScalar, iIntFrameNum);
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


HRESULT CHopperRender::InterpolateFrame(BYTE* pInBuffer, BYTE* pOutBuffer, double dScalar, int iIntFrameNum) {
    // Get the image properties from the BITMAPINFOHEADER
    AM_MEDIA_TYPE* pType = &m_pInput->CurrentMediaType();
    VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)pType->pbFormat;
    ASSERT(pvi);
    const unsigned int dimX = pvi->bmiHeader.biWidth;
    const unsigned int dimY = pvi->bmiHeader.biHeight;

    // Initialize the GPU Arrays if they haven't been initialized yet
    if (!m_ofcOpticalFlowCalc.isInitialized()) {
        m_ofcOpticalFlowCalc.init(dimY, dimX);
    }

    // Either fill the A or B frame with the new data, so that
    // we always have the current frame and the previous frame
    if (iIntFrameNum == 0) {
        if (m_bBisNewest) {
            m_ofcOpticalFlowCalc.updateFrame1(pInBuffer);
        } else {
            m_ofcOpticalFlowCalc.updateFrame2(pInBuffer);
        }
    }

    // Calculate the optical flow
    //if (iIntFrameNum == 0) {
    //	m_ofcOpticalFlowCalc.calculateOpticalFlow();
	//}

    // Warp the frame
    m_ofcOpticalFlowCalc.warpFrame(dScalar, m_bBisNewest);

    // Blend the frames together
	//m_ofcOpticalFlowCalc.blendFrames(dScalar, m_bBisNewest);

    // Download the result to the Output Sample
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

    WRITEOUT(m_bActivated)
    WRITEOUT(m_iNumSteps)
    WRITEOUT(m_iMaxOffsetDivider)
    WRITEOUT(m_bIntNeeded)
    WRITEOUT(m_rtAvgSourceFrameTime)

    return NOERROR;
}


// Likewise overriden to restore our state from a stream
HRESULT CHopperRender::ReadFromStream(IStream* pStream) {
    HRESULT hr;

    READIN(m_bActivated)
    READIN(m_iNumSteps)
    READIN(m_iMaxOffsetDivider)
    READIN(m_bIntNeeded)
    READIN(m_rtAvgSourceFrameTime)

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


// Return the current settings selected
STDMETHODIMP CHopperRender::get_Settings(bool* pbActivated, int* piNumSteps, int* piMaxOffsetDivider, int* iIntActiveState, double* dSourceFPS) {
    CAutoLock cAutolock(&m_csHopperRenderLock);
    CheckPointer(pbActivated, E_POINTER)
    CheckPointer(piNumSteps, E_POINTER)
    CheckPointer(piMaxOffsetDivider, E_POINTER)
    CheckPointer(iIntActiveState, E_POINTER)
    CheckPointer(dSourceFPS, E_POINTER)

    *pbActivated = m_bActivated;
    *piNumSteps = m_iNumSteps;
    *piMaxOffsetDivider = m_iMaxOffsetDivider;
    if (m_bActivated && m_bIntNeeded) {
		*iIntActiveState = 2;
    } else if (m_bActivated && !m_bIntNeeded) {
		*iIntActiveState = 1;
    } else {
        *iIntActiveState = 0;
    }
    *dSourceFPS = 10000000.0 / static_cast<double>(m_rtAvgSourceFrameTime);

    return NOERROR;
}


// Set the settings
STDMETHODIMP CHopperRender::put_Settings(bool bActivated, int iNumSteps, int iMaxOffsetDivider) {
    CAutoLock cAutolock(&m_csHopperRenderLock);

    m_bActivated = bActivated;
    m_iNumSteps = iNumSteps;
    m_iMaxOffsetDivider = iMaxOffsetDivider;

    SetDirty(TRUE);
    return NOERROR;
}