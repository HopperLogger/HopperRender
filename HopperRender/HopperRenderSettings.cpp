#include <windows.h>
#include <windowsx.h>
#include <streams.h>
#include <commctrl.h>
#include <olectl.h>
#include <memory.h>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <tchar.h>
#include <iomanip>
#include <sstream>
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
	m_iFrameOutput(2),
	m_iNumIterations(0),
	m_iFrameBlurKernelSize(20),
	m_iFlowBlurKernelSize(14),
	m_iIntActiveState(0),
	m_dSourceFPS(0.0),
	m_iNumSteps(0),
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
			return 1;
		}
	}

	// Get the current settings
	int iDimX;
	int iDimY;
	int iLowDimX;
	int iLowDimY;
	m_pSettingsInterface->get_Settings(&m_bActivated, &m_iFrameOutput, &m_iNumIterations, &m_iFrameBlurKernelSize, &m_iFlowBlurKernelSize,
	                                   &m_iIntActiveState, &m_dSourceFPS, &m_iNumSteps, &iDimX, &iDimY, &iLowDimX, &iLowDimY);

	// Update the filter active status
	if (m_iIntActiveState == 3) {
		SetDlgItemText(m_Dlg, IDC_INTACTIVE, TEXT("Too Slow"));
	} else if (m_iIntActiveState == 2) {
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

	// Update the number of steps
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iNumSteps);
	SetDlgItemText(m_Dlg, IDC_NUMSTEPS, sz);

	// Update the frame resolution
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d x %d\0"), iDimX, iDimY);
	SetDlgItemText(m_Dlg, IDC_FRAMERES, sz);

	// Update the calculation resolution
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d x %d\0"), iLowDimX, iLowDimY);
	SetDlgItemText(m_Dlg, IDC_CALCRES, sz);

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
	int iDimX;
	int iDimY;
	int iLowDimX;
	int iLowDimY;
	CheckPointer(m_pSettingsInterface, E_FAIL);
	m_pSettingsInterface->get_Settings(&m_bActivated, &m_iFrameOutput, &m_iNumIterations, &m_iFrameBlurKernelSize, &m_iFlowBlurKernelSize,
	                                   &m_iIntActiveState, &m_dSourceFPS, &m_iNumSteps, &iDimX, &iDimY, &iLowDimX, &iLowDimY);

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

	// Set the initial FrameBlurKernelSize
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iFrameBlurKernelSize);
	Edit_SetText(GetDlgItem(m_Dlg, IDC_FRAMEBLURKERNEL), sz);

	// Set the initial FlowBlurKernelSize
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iFlowBlurKernelSize);
	Edit_SetText(GetDlgItem(m_Dlg, IDC_FLOWBLURKERNEL), sz);

	// Set the initial NumIterations
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iNumIterations);
	Edit_SetText(GetDlgItem(m_Dlg, IDC_NUMITS), sz);

	// Check the appropriate radio button
	if (m_bActivated) {
		CheckRadioButton(m_Dlg, IDC_ON, IDC_OFF, IDC_ON);
	} else {
		CheckRadioButton(m_Dlg, IDC_ON, IDC_OFF, IDC_OFF);
	}

	// Update the selected frame output
	if (m_iFrameOutput == 0) {
		CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLURREDFRAMES, IDC_WARPEDFRAME12);
	} else if (m_iFrameOutput == 1) {
		CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLURREDFRAMES, IDC_WARPEDFRAME21);
	} else if (m_iFrameOutput == 2) {
		CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLURREDFRAMES, IDC_BLENDEDFRAME);
	} else if (m_iFrameOutput == 3) {
		CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLURREDFRAMES, IDC_HSVFLOW);
	} else {
		CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLURREDFRAMES, IDC_BLURREDFRAMES);
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
	if (m_iFrameBlurKernelSize > 32) {
		m_iFrameBlurKernelSize = 32;
		TCHAR sz[60];
		(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iFrameBlurKernelSize);
		Edit_SetText(GetDlgItem(m_Dlg, IDC_FRAMEBLURKERNEL), sz);
	}
	m_pSettingsInterface->put_Settings(m_bActivated, m_iFrameOutput, m_iNumIterations, m_iFrameBlurKernelSize, m_iFlowBlurKernelSize);
	if (saveSettings() != S_OK) {
		return E_FAIL;
	}

	return NOERROR;
}

// Get the values from the controls
void CHopperRenderSettings::GetControlValues() {
	TCHAR sz[STR_MAX_LENGTH];
	int tmp1, tmp2, tmp3;

	// Get the number of iterations
	Edit_GetText(GetDlgItem(m_Dlg, IDC_NUMITS), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	char szANSI[STR_MAX_LENGTH];
	int rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
	tmp1 = atoi(szANSI);
#else
    tmp1 = atoi(sz);
#endif

	// Get the frame blur kernel size
	Edit_GetText(GetDlgItem(m_Dlg, IDC_FRAMEBLURKERNEL), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
	tmp2 = atoi(szANSI);
#else
    tmp2 = atoi(sz);
#endif

	// Get the flow blur kernel size
	Edit_GetText(GetDlgItem(m_Dlg, IDC_FLOWBLURKERNEL), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
	tmp3 = atoi(szANSI);
#else
    tmp3 = atoi(sz);
#endif

	m_iNumIterations = tmp1;
	m_iFrameBlurKernelSize = tmp2;
	m_iFlowBlurKernelSize = tmp3;

	// Find whether the filter is activated or not
	if (IsDlgButtonChecked(m_Dlg, IDC_ON)) {
		m_bActivated = true;
	} else {
		m_bActivated = false;
	}

	// Find the frame output
	if (IsDlgButtonChecked(m_Dlg, IDC_WARPEDFRAME12)) {
		m_iFrameOutput = 0;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_WARPEDFRAME21)) {
		m_iFrameOutput = 1;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_BLENDEDFRAME)) {
		m_iFrameOutput = 2;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_HSVFLOW)) {
		m_iFrameOutput = 3;
	} else {
		m_iFrameOutput = 4;
	}
}

// Saves the settings to the registry
HRESULT CHopperRenderSettings::saveSettings() {
	HKEY hKey;
    LPCWSTR subKey = L"SOFTWARE\\HopperRender";

    // Create or open the registry key
    LONG result0 = RegCreateKeyEx(HKEY_CURRENT_USER, subKey, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, NULL, &hKey, NULL);
    
    if (result0 == ERROR_SUCCESS) {
        // Save activated state
        LONG result1 = RegSetValueEx(hKey, L"Activated", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_bActivated), sizeof(DWORD));

		// Save Frame Output
		LONG result2 = RegSetValueEx(hKey, L"FrameOutput", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iFrameOutput), sizeof(DWORD));

		// Save the number of iterations
		LONG result3 = RegSetValueEx(hKey, L"NumIterations", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iNumIterations), sizeof(DWORD));

		// Save the flow blur kernel size
		LONG result4 = RegSetValueEx(hKey, L"FrameBlurKernelSize", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iFrameBlurKernelSize), sizeof(DWORD));

		// Save the flow blur kernel size
		LONG result5 = RegSetValueEx(hKey, L"FlowBlurKernelSize", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iFlowBlurKernelSize), sizeof(DWORD));

		RegCloseKey(hKey); // Close the registry key

		// Check for errors
        if (result1 || result2 || result3 || result4 || result5) {
			return E_FAIL;
        } else {
            return S_OK;
        }

    } else {
        return E_FAIL;
    }
}