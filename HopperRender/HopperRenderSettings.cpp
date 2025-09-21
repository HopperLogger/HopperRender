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
	CBasePropertyPage(NAME("HopperRender Settings"), pUnk, IDD_HopperRenderSettings, IDS_TITLE),
	m_bActivated(false),
	m_iFrameOutput(BlendedFrame),
	m_iDeltaScalar(8),
	m_iNeighborScalar(6),
	m_iBlackLevel(0),
	m_iWhiteLevel(255),
	m_iIntActiveState(Active),
	m_dSourceFPS(0.0),
	m_dTargetFPS(0.0),
    m_bUseDisplayFPS(true),
	m_dOFCCalcTime(0.0),
	m_dWarpCalcTime(0.0),
	m_bIsInitialized(false),
	m_pSettingsInterface(nullptr) {
	ASSERT(phr);
}

// Handles the messages for our property window
INT_PTR CHopperRenderSettings::OnReceiveMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	switch (uMsg) {
		case WM_COMMAND: {
			TCHAR sz[STR_MAX_LENGTH];
			const int nID = LOWORD(wParam);
			int action = HIWORD(wParam);
			if (m_bIsInitialized) {
				m_bDirty = true;
				if (m_pPageSite) {
					m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
				}
				// Check if the defaults button was pressed
				if (action == BN_CLICKED && nID == IDC_DEFAULTS) {
					m_bActivated = true;
					m_iFrameOutput = BlendedFrame;
					m_dTargetFPS = 0.0; // Uses the display fps
					m_iDeltaScalar = DEFAULT_DELTA_SCALAR;
					m_iNeighborScalar = DEFAULT_NEIGHBOR_SCALAR;
					m_iBlackLevel = DEFAULT_BLACK_LEVEL;
					m_iWhiteLevel = DEFAULT_WHITE_LEVEL;
					CheckRadioButton(m_Dlg, IDC_ON, IDC_OFF, IDC_ON); 
					CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_SIDEBYSIDE2, IDC_BLENDEDFRAME);
					(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f\0"), m_dTargetFPS);
					SetDlgItemText(m_Dlg, IDC_TARGETFPS, sz);
					(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iDeltaScalar);
					Edit_SetText(GetDlgItem(m_Dlg, IDC_DELTASCALAR), sz);
					(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iNeighborScalar);
					Edit_SetText(GetDlgItem(m_Dlg, IDC_NEIGHBORSCALAR), sz);
					(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iBlackLevel);
					Edit_SetText(GetDlgItem(m_Dlg, IDC_BLACKLEVEL), sz);
					(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iWhiteLevel);
					Edit_SetText(GetDlgItem(m_Dlg, IDC_WHITELEVEL), sz);
					CheckDlgButton(m_Dlg, IDC_DEFAULTS, BST_UNCHECKED);
					CheckDlgButton(m_Dlg, IDC_USEDISPLAYFPS, BST_CHECKED);
				}
			}
			break;
		}
		case WM_INITDIALOG:
		case WM_TIMER: {
			// Get the current settings
			int iDimX;
			int iDimY;
			int iLowDimX;
			int iLowDimY;
			int frameOutput;
			int activeState;
			double currentFPS = m_dSourceFPS;
			m_pSettingsInterface->GetCurrentSettings(&m_bActivated, &frameOutput, &m_dTargetFPS, &m_bUseDisplayFPS, &m_iDeltaScalar, &m_iNeighborScalar, &m_iBlackLevel, &m_iWhiteLevel,
													 &activeState, &m_dSourceFPS, &m_dOFCCalcTime, &m_dAVGOFCCalcTime, &m_dPeakOFCCalcTime, &m_dWarpCalcTime, &iDimX, &iDimY, &iLowDimX, &iLowDimY);
			m_iFrameOutput = static_cast<FrameOutput>(frameOutput);
			m_iIntActiveState = static_cast<ActiveState>(activeState);

			// Update the status for every frame
			if (currentFPS != m_dSourceFPS) {
				KillTimer(m_Dlg, 1);
				int delay = (1.0 / m_dSourceFPS) * 1000.0;
				SetTimer(m_Dlg, 1, delay, nullptr);
			}

			// Update the filter active status
			switch (m_iIntActiveState) {
				case Deactivated:
					SetDlgItemText(m_Dlg, IDC_INTACTIVE, TEXT("Deactivated"));
					break;
				case NotNeeded:
					SetDlgItemText(m_Dlg, IDC_INTACTIVE, TEXT("Not Needed"));
					break;
				case Active:
					SetDlgItemText(m_Dlg, IDC_INTACTIVE, TEXT("Active"));
					break;
				case TooSlow:
					SetDlgItemText(m_Dlg, IDC_INTACTIVE, TEXT("Too Slow"));
					break;
			}

			// Update the source frames per second
			TCHAR sz[60];
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f fps\0"), m_dSourceFPS);
			SetDlgItemText(m_Dlg, IDC_SOURCEFPS, sz);

			// Update the OFC Calc Time
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f ms\0"), m_dOFCCalcTime);
			SetDlgItemText(m_Dlg, IDC_OFCCALCTIME, sz);

			// Update the AVG OFC Calc Time
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f ms\0"), m_dAVGOFCCalcTime);
			SetDlgItemText(m_Dlg, IDC_AVGOFCCALCTIME, sz);

			// Update the Peak OFC Calc Time
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f ms\0"), m_dPeakOFCCalcTime);
			SetDlgItemText(m_Dlg, IDC_PEAKOFCCALCTIME, sz);

			// Update the Warp Calc Time
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f ms\0"), m_dWarpCalcTime);
			SetDlgItemText(m_Dlg, IDC_WARPCALCTIME, sz);

			// Update the frame resolution
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d x %d\0"), iDimX, iDimY);
			SetDlgItemText(m_Dlg, IDC_FRAMERES, sz);

			// Update the calculation resolution
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d x %d\0"), iLowDimX, iLowDimY);
			SetDlgItemText(m_Dlg, IDC_CALCRES, sz);
		}
	}

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
	int frameOutput;
	int activeState;
	CheckPointer(m_pSettingsInterface, E_FAIL);
	m_pSettingsInterface->GetCurrentSettings(&m_bActivated, &frameOutput, &m_dTargetFPS, &m_bUseDisplayFPS, &m_iDeltaScalar, &m_iNeighborScalar, &m_iBlackLevel, &m_iWhiteLevel,
	                                         &activeState, &m_dSourceFPS, &m_dOFCCalcTime, &m_dAVGOFCCalcTime, &m_dPeakOFCCalcTime, &m_dWarpCalcTime, &iDimX, &iDimY, &iLowDimX, &iLowDimY);
	m_iFrameOutput = static_cast<FrameOutput>(frameOutput);
	m_iIntActiveState = static_cast<ActiveState>(activeState);

	m_bIsInitialized = false;
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

	// Set the initial activated state
	if (m_bActivated) {
		CheckRadioButton(m_Dlg, IDC_ON, IDC_OFF, IDC_ON);
	} else {
		CheckRadioButton(m_Dlg, IDC_ON, IDC_OFF, IDC_OFF);
	}

	// Update the selected frame output
	switch (m_iFrameOutput) {
		case WarpedFrame12:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_SIDEBYSIDE2, IDC_WARPEDFRAME12);
			break;
		case WarpedFrame21:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_SIDEBYSIDE2, IDC_WARPEDFRAME21);
			break;
		case BlendedFrame:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_SIDEBYSIDE2, IDC_BLENDEDFRAME);
			break;
		case HSVFlow:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_SIDEBYSIDE2, IDC_HSVFLOW);
			break;
		case GreyFlow:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_SIDEBYSIDE2, IDC_GREYFLOW);
			break;
		case SideBySide1:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_SIDEBYSIDE2, IDC_SIDEBYSIDE1);
			break;
		case SideBySide2:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_SIDEBYSIDE2, IDC_SIDEBYSIDE2);
			break;
	}

	// Set the initial Use Display FPS checkbox
	if (m_bUseDisplayFPS) {
		CheckDlgButton(m_Dlg, IDC_USEDISPLAYFPS, BST_CHECKED);
	} else {
		CheckDlgButton(m_Dlg, IDC_USEDISPLAYFPS, BST_UNCHECKED);
	}

	// Set the initial Target FPS
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f\0"), m_dTargetFPS);
	SetDlgItemText(m_Dlg, IDC_TARGETFPS, sz);

	// Set the initial Delta Scalar
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iDeltaScalar);
	Edit_SetText(GetDlgItem(m_Dlg, IDC_DELTASCALAR), sz);

	// Set the initial Neighbor Scalar
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iNeighborScalar);
	Edit_SetText(GetDlgItem(m_Dlg, IDC_NEIGHBORSCALAR), sz);

	// Set the initial Black Level
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iBlackLevel);
	Edit_SetText(GetDlgItem(m_Dlg, IDC_BLACKLEVEL), sz);

	// Set the initial White Level
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iWhiteLevel);
	Edit_SetText(GetDlgItem(m_Dlg, IDC_WHITELEVEL), sz);

	// Update the status for every frame
	int delay = (1.0 / m_dSourceFPS) * 1000.0;
	SetTimer(m_Dlg, 1, delay, nullptr);
	m_bIsInitialized = true;

	return NOERROR;
}

// We are being deactivated
HRESULT CHopperRenderSettings::OnDeactivate() {
	ASSERT(m_pSettingsInterface);

	m_bIsInitialized = false;
	GetControlValues();

	return NOERROR;
}

// Validate the parameter values
void CHopperRenderSettings::ValidateParameter(int& parameter, int maxValue, int controlId) {
    if (parameter < 0) {
        parameter = 0;
    } else if (parameter > maxValue) {
        parameter = maxValue;
    }
    TCHAR sz[60];
    (void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), parameter);
    Edit_SetText(GetDlgItem(m_Dlg, controlId), sz);
}

// Apply any changes so far made
HRESULT CHopperRenderSettings::OnApplyChanges() {
	GetControlValues();

	CheckPointer(m_pSettingsInterface, E_POINTER)

	ValidateParameter(m_iDeltaScalar, 10, IDC_DELTASCALAR);
	ValidateParameter(m_iNeighborScalar, 10, IDC_NEIGHBORSCALAR);
	ValidateParameter(m_iBlackLevel, 255, IDC_BLACKLEVEL);
	ValidateParameter(m_iWhiteLevel, 255, IDC_WHITELEVEL);

	// Tell the filter about the new settings
	m_pSettingsInterface->UpdateUserSettings(m_bActivated, m_iFrameOutput, m_dTargetFPS, m_bUseDisplayFPS, m_iDeltaScalar, m_iNeighborScalar, m_iBlackLevel, m_iWhiteLevel);

	// Save the settings to the registry
	if (saveSettings() != S_OK) {
		return E_FAIL;
	}

	// Show the settings the filter accepted
	int iDimX;
	int iDimY;
	int iLowDimX;
	int iLowDimY;
	int frameOutput;
	int activeState;
	CheckPointer(m_pSettingsInterface, E_FAIL);
	m_pSettingsInterface->GetCurrentSettings(&m_bActivated, &frameOutput, &m_dTargetFPS, &m_bUseDisplayFPS, &m_iDeltaScalar, &m_iNeighborScalar, &m_iBlackLevel, &m_iWhiteLevel,
	                                         &activeState, &m_dSourceFPS, &m_dOFCCalcTime, &m_dAVGOFCCalcTime, &m_dPeakOFCCalcTime, &m_dWarpCalcTime, &iDimX, &iDimY, &iLowDimX, &iLowDimY);
	TCHAR sz[60];
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f\0"), m_dTargetFPS);
	SetDlgItemText(m_Dlg, IDC_TARGETFPS, sz);

	return NOERROR;
}

// Get the values from the controls
void CHopperRenderSettings::GetControlValues() {
	TCHAR sz[STR_MAX_LENGTH];
    double tmp1;
	int tmp2, tmp3, tmp4, tmp5;

	// Find whether the filter is activated or not
	m_bActivated = IsDlgButtonChecked(m_Dlg, IDC_ON);

	// Find the frame output
	if (IsDlgButtonChecked(m_Dlg, IDC_WARPEDFRAME12)) {
		m_iFrameOutput = WarpedFrame12;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_WARPEDFRAME21)) {
		m_iFrameOutput = WarpedFrame21;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_BLENDEDFRAME)) {
		m_iFrameOutput = BlendedFrame;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_HSVFLOW)) {
		m_iFrameOutput = HSVFlow;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_GREYFLOW)) {
		m_iFrameOutput = GreyFlow;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_SIDEBYSIDE1)) {
		m_iFrameOutput = SideBySide1;
	} else {
		m_iFrameOutput = SideBySide2;
	}

	// Check if the use display fps checkbox is checked
	m_bUseDisplayFPS = IsDlgButtonChecked(m_Dlg, IDC_USEDISPLAYFPS);

	// Get the target fps
	Edit_GetText(GetDlgItem(m_Dlg, IDC_TARGETFPS), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	char szANSI[STR_MAX_LENGTH];
	int rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH,
				     nullptr, nullptr);
	tmp1 = atof(szANSI);
#else
	tmp1 = atof(sz);
#endif

	// Get the delta scalar value
	Edit_GetText(GetDlgItem(m_Dlg, IDC_DELTASCALAR), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
	tmp2 = atoi(szANSI);
#else
    tmp2 = atoi(sz);
#endif

	// Get the neighbor scalar value
	Edit_GetText(GetDlgItem(m_Dlg, IDC_NEIGHBORSCALAR), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
	tmp3 = atoi(szANSI);
#else
    tmp3 = atoi(sz);
#endif

	// Get the black level
	Edit_GetText(GetDlgItem(m_Dlg, IDC_BLACKLEVEL), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
	tmp4 = atoi(szANSI);
#else
    tmp4 = atoi(sz);
#endif

	// Get the white level
	Edit_GetText(GetDlgItem(m_Dlg, IDC_WHITELEVEL), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
	tmp5 = atoi(szANSI);
#else
    tmp5 = atoi(sz);
#endif

	m_dTargetFPS = tmp1;
	m_iDeltaScalar = tmp2;
	m_iNeighborScalar = tmp3;
	m_iBlackLevel = tmp4;
	m_iWhiteLevel = tmp5;
}

// Saves the settings to the registry
HRESULT CHopperRenderSettings::saveSettings() {
	HKEY hKey;
    LPCWSTR subKey = L"SOFTWARE\\HopperRender";

    // Create or open the registry key
    LONG result0 = RegCreateKeyEx(HKEY_CURRENT_USER, subKey, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, NULL, &hKey, NULL);
    
    if (result0 == ERROR_SUCCESS) {
        // Save activated state
		DWORD activated = m_bActivated ? 1 : 0;
        LONG result1 = RegSetValueEx(hKey, L"Activated", 0, REG_DWORD, reinterpret_cast<BYTE*>(&activated), sizeof(DWORD));

		// Save Frame Output
		LONG result2 = RegSetValueEx(hKey, L"FrameOutput", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iFrameOutput), sizeof(DWORD));

		// Save the target fps
		LONG result3 = RegSetValueEx(hKey, L"TargetFPS", 0, REG_BINARY, reinterpret_cast<const BYTE*>(&m_dTargetFPS), sizeof(double));

		// Save the use display fps flag
		DWORD useDisplayFPS = m_bUseDisplayFPS ? 1 : 0;
		LONG result4 = RegSetValueEx(hKey, L"Use Display FPS", 0, REG_DWORD, reinterpret_cast<BYTE*>(&useDisplayFPS), sizeof(DWORD));

		// Save the delta scalar
		LONG result5 = RegSetValueEx(hKey, L"DeltaScalar", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iDeltaScalar), sizeof(DWORD));

		// Save the neighbor scalar
		LONG result6 = RegSetValueEx(hKey, L"NeighborScalar", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iNeighborScalar), sizeof(DWORD));

		// Save the black level
		LONG result7 = RegSetValueEx(hKey, L"BlackLevel", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iBlackLevel), sizeof(DWORD));

		// Save the white level
		LONG result8 = RegSetValueEx(hKey, L"WhiteLevel", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iWhiteLevel), sizeof(DWORD));
		
		RegCloseKey(hKey); // Close the registry key

		// Check for errors
        if (result1 || result2 || result3 || result4 || result5 || result6 || result7 || result8) {
			return E_FAIL;
        } else {
            return S_OK;
        }

    } else {
        return E_FAIL;
    }
}
