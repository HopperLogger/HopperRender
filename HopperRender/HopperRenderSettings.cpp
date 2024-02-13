#include <windows.h>
#include <windowsx.h>
#include <streams.h>
#include <commctrl.h>
#include <olectl.h>
#include <memory.h>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <tchar.h>
#include "resource.h"
#include "uids.h"
#include "iEZ.h"
#include "HopperRender.h"
#include "HopperRenderSettings.h"


// Used by the DirectShow base classes to create instances
CUnknown* CHopperRenderSettings::CreateInstance(LPUNKNOWN lpunk, HRESULT* phr) {
    ASSERT(phr);

    CUnknown* punk = new CHopperRenderSettings(lpunk, phr);

    if (punk == nullptr) {
        if (phr) {
            *phr = E_OUTOFMEMORY;
        }
    }

    return punk;
}


// Constructor
CHopperRenderSettings::CHopperRenderSettings(LPUNKNOWN pUnk, HRESULT* phr) :
    CBasePropertyPage(NAME("HopperRender Settings"), pUnk,
        IDD_HopperRenderSettings, IDS_TITLE),
    m_bIsInitialized(FALSE),
    m_bActivated(FALSE),
    m_iNumSteps(40),
    m_iMaxOffsetDivider(192),
    m_iIntActiveState(0),
    m_dSourceFPS(0.0),
    m_pSettingsInterface(nullptr) {

    ASSERT(phr);
}


// Handles the messages for our property window
INT_PTR CHopperRenderSettings::OnReceiveMessage(HWND hwnd,
    UINT uMsg,
    WPARAM wParam,
    LPARAM lParam) {

    switch (uMsg) {
	    case WM_COMMAND: {
	        if (m_bIsInitialized) {
	            m_bDirty = TRUE;
	            if (m_pPageSite) {
	                m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
	            }
	        }
	        return (LRESULT)1;
	    }
    }

    // Get the current settings
    m_pSettingsInterface->get_Settings(&m_bActivated, &m_iNumSteps, &m_iMaxOffsetDivider, &m_iIntActiveState, &m_dSourceFPS);

    // Update the effect active status
    if (m_iIntActiveState == 2) {
        SetDlgItemText(m_Dlg, IDC_INTACTIVE, TEXT("Active"));
    } else if (m_iIntActiveState == 1) {
        SetDlgItemText(m_Dlg, IDC_INTACTIVE, TEXT("Not Needed"));
    } else {
		SetDlgItemText(m_Dlg, IDC_INTACTIVE, TEXT("Deactivated"));
	}

    // Update the source frames per second
    TCHAR sz[60];
    (void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f fps\0"), m_dSourceFPS);
    SetDlgItemText(m_Dlg, IDC_SOURCEFPS, sz);

    return CBasePropertyPage::OnReceiveMessage(hwnd, uMsg, wParam, lParam);
}


// Called when we connect to a transform filter
HRESULT CHopperRenderSettings::OnConnect(IUnknown* pUnknown) {
    CheckPointer(pUnknown, E_POINTER)
    ASSERT(m_pSettingsInterface == NULL);

    HRESULT hr = pUnknown->QueryInterface(IID_SettingsInterface, (void**)&m_pSettingsInterface);
    if (FAILED(hr)) {
        return E_NOINTERFACE;
    }

    // Get the initial settings
    CheckPointer(m_pSettingsInterface, E_FAIL);
    m_pSettingsInterface->get_Settings(&m_bActivated, &m_iNumSteps, &m_iMaxOffsetDivider, &m_iIntActiveState, &m_dSourceFPS);

    m_bIsInitialized = FALSE;
    return NOERROR;
}


// Likewise called when we disconnect from a filter
HRESULT CHopperRenderSettings::OnDisconnect() {
    // Release of Interface after setting the appropriate old settings
    if (m_pSettingsInterface) {
        m_pSettingsInterface->Release();
        m_pSettingsInterface = nullptr;
    }
    return NOERROR;
}


// We are being activated
HRESULT CHopperRenderSettings::OnActivate() {
    TCHAR sz[60];

    // Set the initial MaxOffsetDivider
    (void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iMaxOffsetDivider);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_MAXOFFSETDIV), sz);

    // Set the initial NumSteps
    (void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iNumSteps);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_NUMSTEPS), sz);

    // Check the appropriate radio button
    if (m_bActivated) {
		CheckRadioButton(m_Dlg, IDC_ON, IDC_OFF, IDC_ON);
    } else {
		CheckRadioButton(m_Dlg, IDC_ON, IDC_OFF, IDC_OFF);
	}
    m_bIsInitialized = TRUE;

    return NOERROR;
}


// We are being deactivated
HRESULT CHopperRenderSettings::OnDeactivate() {
    ASSERT(m_pSettingsInterface);

    m_bIsInitialized = FALSE;
    GetControlValues();

    return NOERROR;
}


// Apply any changes so far made
HRESULT CHopperRenderSettings::OnApplyChanges() {
    GetControlValues();

    CheckPointer(m_pSettingsInterface, E_POINTER)
        m_pSettingsInterface->put_Settings(m_bActivated, m_iNumSteps, m_iMaxOffsetDivider);

    return NOERROR;
}

// Get the values from the controls
void CHopperRenderSettings::GetControlValues() {
    TCHAR sz[STR_MAX_LENGTH];
    int tmp1, tmp2;

    // Get the max offset divider
    Edit_GetText(GetDlgItem(m_Dlg, IDC_MAXOFFSETDIV), sz, STR_MAX_LENGTH);

#ifdef UNICODE
    // Convert Multibyte string to ANSI
    char szANSI[STR_MAX_LENGTH];

    int rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
    tmp2 = atoi(szANSI);
#else
    tmp2 = atoi(sz);
#endif

    // Get the number of steps
    Edit_GetText(GetDlgItem(m_Dlg, IDC_NUMSTEPS), sz, STR_MAX_LENGTH);

#ifdef UNICODE
    // Convert Multibyte string to ANSI
    rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
    tmp1 = atoi(szANSI);
#else
    tmp1 = atoi(sz);
#endif

    m_iNumSteps = tmp1;
    m_iMaxOffsetDivider = tmp2;

    // Find whether the filter is activated or not
    if (IsDlgButtonChecked(m_Dlg, IDC_ON)) {
        m_bActivated = true;
    } else {
		m_bActivated = false;
	}
}