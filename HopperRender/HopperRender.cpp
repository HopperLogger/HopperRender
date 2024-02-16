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
	m_iFrameOutput(2),
	m_iNumIterations(0),
    m_iNumSteps(10),
    m_iMaxOffsetDivider(192),
    m_lBufferRequest(12),
    m_bBisNewest(true),
    m_iFrameCounter(0),
    m_rtCurrStartTime(LONGLONG_MAX),
    m_rtLastStartTime(LONGLONG_MAX),
    m_bIntNeeded(false),
    m_rtAvgSourceFrameTime(0),
	m_rtCurrPlaybackFrameTime(0) {
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
                m_rtAvgSourceFrameTime = pvi->AvgTimePerFrame;
                const unsigned int dimX = pvi->bmiHeader.biWidth;
                const unsigned int dimY = pvi->bmiHeader.biHeight;

                // Initialize the Optical Flow Calculator
                if (!m_ofcOpticalFlowCalc.isInitialized()) {
                    m_ofcOpticalFlowCalc.init(dimY, dimX);
                }
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


// Called when a new segment is started
HRESULT CHopperRender::NewSegment(REFERENCE_TIME tStart, REFERENCE_TIME tStop, double dRate) {
    m_rtCurrPlaybackFrameTime = static_cast<REFERENCE_TIME>(static_cast<double>(m_rtAvgSourceFrameTime) * (1.0 / dRate));
    return __super::NewSegment(tStart, tStop, dRate);
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
    int iNumSamples = 5;

    for (int iIntFrameNum = 0; iIntFrameNum < iNumSamples; ++iIntFrameNum) {
        // Create a new output sample
        if (iIntFrameNum < (iNumSamples - 1)) {
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

        // Set the presentation time for the new output sample
        REFERENCE_TIME rtStartTime, rtEndTime;
        hr = pIn->GetTime(&rtStartTime, &rtEndTime);
        if (FAILED(hr)) {
            return hr;
        }

        // Check if interpolation is necessary
        if (m_bActivated) {
            if (m_rtCurrPlaybackFrameTime > (rtAvgFrameTimeTarget + 833)) {
                m_bIntNeeded = true;
            } else {
                m_bIntNeeded = false;
            }
        } else {
			m_bIntNeeded = false;
		}

        if (m_bIntNeeded) { // Interpolation needed

            if (iIntFrameNum == 0) {
                // Print the original start and end times
                //DebugMessage("ACINT | Start time: " + std::to_string(rtStartTime) + " End time: " + std::to_string(rtEndTime) + " Delta: " + std::to_string(rtEndTime - rtStartTime) + " Start2-Start1: " + std::to_string(rtStartTime - m_rtLastStartTime) + " AVG S2-S1: " + std::to_string(m_rtCurrPlaybackFrameTime));

                // Reset our frame time if necessary and calculate the current number of intermediate frames needed
                if (rtStartTime < m_rtLastStartTime) {
                    m_rtCurrStartTime = rtStartTime;
                    iNumSamples = static_cast<int>(ceil((static_cast<double>(rtEndTime) - static_cast<double>(rtStartTime)) / static_cast<double>(rtAvgFrameTimeTarget)));
                } else {
                    iNumSamples = static_cast<int>(ceil(static_cast<double>((m_rtCurrPlaybackFrameTime + rtStartTime - m_rtCurrStartTime)) / static_cast<double>(rtAvgFrameTimeTarget)));
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
            //DebugMessage("NOINT | Start time: " + std::to_string(rtStartTime) + " End time: " + std::to_string(rtEndTime) + " Delta: " + std::to_string(rtEndTime - rtStartTime) + " Start2-Start1: " + std::to_string(rtStartTime - m_rtLastStartTime) + " AVG S2-S1: " + std::to_string(m_rtCurrPlaybackFrameTime));
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
        //const double dScalar = max(min((static_cast<double>(m_rtCurrStartTime) - static_cast<double>(m_rtLastStartTime)) / static_cast<double>(m_rtCurrPlaybackFrameTime), 1.0), 0.0);
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

        // Interpolate the frame if necessary
        if (m_bIntNeeded) {
            hr = InterpolateFrame(pInBuffer, pOutNewBuffer, dScalar, iIntFrameNum);
            if (FAILED(hr)) {
                return hr;
            }
        } else {
            memcpy(pOutNewBuffer, pInBuffer, lInSize);
        }

        // Deliver the new output sample downstream
        if (iIntFrameNum < (iNumSamples - 1)) {
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
    // Either fill the A or B frame with the new data, so that
    // we always have the current frame and the previous frame
    if (iIntFrameNum == 0) {
        if (m_bBisNewest) {
            m_ofcOpticalFlowCalc.updateFrame1(pInBuffer);
        } else {
            m_ofcOpticalFlowCalc.updateFrame2(pInBuffer);
        }
		m_bBisNewest = !m_bBisNewest;
    }

    // Calculate the optical flow (frame 1 to frame 2)
    if (iIntFrameNum == 0) {
        const double duration = m_ofcOpticalFlowCalc.calculateOpticalFlow(m_iNumIterations, m_iNumSteps, m_iMaxOffsetDivider);
        if ((duration + 10.0) > static_cast<double>(m_rtCurrPlaybackFrameTime) / 10000.0) {
            DebugMessage("Optical Flow Calculation took too long " + std::to_string(duration) + " ms" + " AVG SFT: " + std::to_string(static_cast<double>(m_rtCurrPlaybackFrameTime) / 10000.0) + " NumSteps: " + std::to_string(m_iNumSteps));
            m_iNumSteps -= 1;
            if (m_iNumSteps < 1) {
				m_iNumSteps = 1;
			}
        } else if ((static_cast<double>(m_rtCurrPlaybackFrameTime) / 10000.0) > (duration + 30.0)) {
			DebugMessage("Optical Flow Calculation has capacity " + std::to_string(duration) + " ms" + " AVG SFT: " + std::to_string(static_cast<double>(m_rtCurrPlaybackFrameTime) / 10000.0) + " NumSteps: " + std::to_string(m_iNumSteps));
			m_iNumSteps += 1;
        } else {
			DebugMessage("Optical Flow Calculation took " + std::to_string(duration) + " ms" + " AVG SFT: " + std::to_string(static_cast<double>(m_rtCurrPlaybackFrameTime) / 10000.0) + " NumSteps: " + std::to_string(m_iNumSteps));
		}
    }

    // Flip the flow array to frame 2 to frame 1
    if (m_iFrameOutput == 1 || m_iFrameOutput == 2) {
    	m_ofcOpticalFlowCalc.flipFlow();
	}

    // Warp frame 1 to frame 2
    if (m_iFrameOutput == 0 || m_iFrameOutput == 2) {
        m_ofcOpticalFlowCalc.warpFrame12(dScalar);
    }

    // Warp frame 2 to frame 1
    if (m_iFrameOutput == 1 || m_iFrameOutput == 2) {
        m_ofcOpticalFlowCalc.warpFrame21(dScalar);
    }

    // Blend the frames together
    if (m_iFrameOutput == 2) {
        m_ofcOpticalFlowCalc.blendFrames(dScalar);
    }

    // Download the result to the Output Sample
    if (m_iFrameOutput == 0) {
        m_ofcOpticalFlowCalc.warpedFrame12.download(pOutBuffer);
    } else if (m_iFrameOutput == 1) {
        m_ofcOpticalFlowCalc.warpedFrame21.download(pOutBuffer);
    } else if (m_iFrameOutput == 2) {
        m_ofcOpticalFlowCalc.blendedFrame.download(pOutBuffer);
    } else {
        m_ofcOpticalFlowCalc.downloadFlowAsHSV(pOutBuffer, 1.0, 1.0, 0.4);
    }
    //m_ofcOpticalFlowCalc.imageDeltaArray.download(pOutBuffer);

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
    WRITEOUT(m_iMaxOffsetDivider)
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
    READIN(m_iMaxOffsetDivider)
    READIN(m_bIntNeeded)
    READIN(m_rtCurrPlaybackFrameTime)
    READIN(m_iNumSteps)

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
STDMETHODIMP CHopperRender::get_Settings(bool* pbActivated, int* piFrameOutput, int* piNumIterations, int* piMaxOffsetDivider, int* piIntActiveState, double* pdSourceFPS, int* piNumSteps) {
    CAutoLock cAutolock(&m_csHopperRenderLock);
    CheckPointer(pbActivated, E_POINTER)
    CheckPointer(piFrameOutput, E_POINTER)
    CheckPointer(piNumIterations, E_POINTER)
    CheckPointer(piMaxOffsetDivider, E_POINTER)
    CheckPointer(piIntActiveState, E_POINTER)
    CheckPointer(pdSourceFPS, E_POINTER)
    CheckPointer(piNumSteps, E_POINTER)

    *pbActivated = m_bActivated;
    *piFrameOutput = m_iFrameOutput;
    *piNumIterations = m_iNumIterations;
    *piMaxOffsetDivider = m_iMaxOffsetDivider;
    if (m_bActivated && m_bIntNeeded) {
		*piIntActiveState = 2;
    } else if (m_bActivated && !m_bIntNeeded) {
		*piIntActiveState = 1;
    } else {
        *piIntActiveState = 0;
    }
    *pdSourceFPS = 10000000.0 / static_cast<double>(m_rtCurrPlaybackFrameTime);
    *piNumSteps = m_iNumSteps;

    return NOERROR;
}


// Set the settings
STDMETHODIMP CHopperRender::put_Settings(bool bActivated, int iFrameOutput, int iNumIterations, int iMaxOffsetDivider) {
    CAutoLock cAutolock(&m_csHopperRenderLock);

    m_bActivated = bActivated;
    m_iFrameOutput = iFrameOutput;
    m_iNumIterations = iNumIterations;
    m_iMaxOffsetDivider = iMaxOffsetDivider;

    SetDirty(TRUE);
    return NOERROR;
}