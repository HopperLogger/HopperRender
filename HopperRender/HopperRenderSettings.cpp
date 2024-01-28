#include <windows.h>
#include <windowsx.h>
#include <streams.h>
#include <commctrl.h>
#include <olectl.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <tchar.h>
#include "resource.h"
#include "uids.h"
#include "iEZ.h"
#include "HopperRender.h"
#include "HopperRenderSettings.h"


//
// CreateInstance
//
// Used by the DirectShow base classes to create instances
//
CUnknown *CHopperRenderSettings::CreateInstance(LPUNKNOWN lpunk, HRESULT *phr)
{
    ASSERT(phr);

    CUnknown *punk = new CHopperRenderSettings(lpunk, phr);

    if (punk == NULL) {
        if (phr)
        	*phr = E_OUTOFMEMORY;
    }

    return punk;

} // CreateInstance


//
// Constructor
//
CHopperRenderSettings::CHopperRenderSettings(LPUNKNOWN pUnk, HRESULT *phr) :
    CBasePropertyPage(NAME("HopperRender Settings"), pUnk,
                      IDD_HopperRenderSettings, IDS_TITLE),
    m_pIPEffect(NULL),
    m_bIsInitialized(FALSE)
{
    ASSERT(phr);

} // (Constructor)


//
// OnReceiveMessage
//
// Handles the messages for our property window
//
INT_PTR CHopperRenderSettings::OnReceiveMessage(HWND hwnd,
                                          UINT uMsg,
                                          WPARAM wParam,
                                          LPARAM lParam)
{
    switch (uMsg)
    {
        case WM_COMMAND:
        {
            if (m_bIsInitialized)
            {
                m_bDirty = TRUE;
                if (m_pPageSite)
                {
                    m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
                }
            }
            return (LRESULT) 1;
        }

    }

    return CBasePropertyPage::OnReceiveMessage(hwnd,uMsg,wParam,lParam);

} // OnReceiveMessage


//
// OnConnect
//
// Called when we connect to a transform filter
//
HRESULT CHopperRenderSettings::OnConnect(IUnknown *pUnknown)
{
    CheckPointer(pUnknown,E_POINTER);
    ASSERT(m_pIPEffect == NULL);

    HRESULT hr = pUnknown->QueryInterface(IID_IIPEffect, (void **) &m_pIPEffect);
    if (FAILED(hr)) {
        return E_NOINTERFACE;
    }

    // Get the initial image FX property
    CheckPointer(m_pIPEffect,E_FAIL);
    m_pIPEffect->get_IPEffect(&m_effect, &m_start, &m_length);

    m_bIsInitialized = FALSE ;
    return NOERROR;

} // OnConnect


//
// OnDisconnect
//
// Likewise called when we disconnect from a filter
//
HRESULT CHopperRenderSettings::OnDisconnect()
{
    // Release of Interface after setting the appropriate old effect value
    if(m_pIPEffect)
    {
        m_pIPEffect->Release();
        m_pIPEffect = NULL;
    }
    return NOERROR;

} // OnDisconnect


//
// OnActivate
//
// We are being activated
//
HRESULT CHopperRenderSettings::OnActivate()
{
    TCHAR sz[60];

    (void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%f\0"), m_length);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_LENGTH), sz);

    (void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%f\0"), m_start);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_START), sz);

    CheckRadioButton(m_Dlg, IDC_EMBOSS, IDC_NONE, m_effect);
    m_bIsInitialized = TRUE;

    return NOERROR;

} // OnActivate


//
// OnDeactivate
//
// We are being deactivated
//
HRESULT CHopperRenderSettings::OnDeactivate(void)
{
    ASSERT(m_pIPEffect);

    m_bIsInitialized = FALSE;
    GetControlValues();

    return NOERROR;

} // OnDeactivate


//
// OnApplyChanges
//
// Apply any changes so far made
//
HRESULT CHopperRenderSettings::OnApplyChanges()
{
    GetControlValues();

    CheckPointer(m_pIPEffect,E_POINTER);
    m_pIPEffect->put_IPEffect(m_effect, m_start, m_length);

    return NOERROR;

} // OnApplyChanges


void CHopperRenderSettings::GetControlValues()
{
    TCHAR sz[STR_MAX_LENGTH];
    REFTIME tmp1, tmp2 ;

    // Get the start and effect times
    Edit_GetText(GetDlgItem(m_Dlg, IDC_LENGTH), sz, STR_MAX_LENGTH);

#ifdef UNICODE
    // Convert Multibyte string to ANSI
    char szANSI[STR_MAX_LENGTH];

    int rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, NULL, NULL);
    tmp2 = COARefTime(atof(szANSI));
#else
    tmp2 = COARefTime(atof(sz));
#endif

    Edit_GetText(GetDlgItem(m_Dlg, IDC_START), sz, STR_MAX_LENGTH);

#ifdef UNICODE
    // Convert Multibyte string to ANSI
    rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, NULL, NULL);
    tmp1 = COARefTime(atof(szANSI));
#else
    tmp1 = COARefTime(atof(sz));
#endif

    // Quick validation of the fields

    if (tmp1 >= 0 && tmp2 >= 0) {
        m_start  = tmp1;
        m_length = tmp2;
    }

    // Find which special effect we have selected

    for (int i = IDC_EMBOSS; i <= IDC_NONE; i++) 
    {
        if (IsDlgButtonChecked(m_Dlg, i)) 
        {
            m_effect = i;
            break;
        }
    }
}
