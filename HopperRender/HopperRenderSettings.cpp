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
	m_iDeltaScalar(DEFAULT_DELTA_SCALAR),
	m_iNeighborScalar(DEFAULT_NEIGHBOR_SCALAR),
	m_iBlackLevel(DEFAULT_BLACK_LEVEL),
	m_iWhiteLevel(DEFAULT_WHITE_LEVEL),
	m_iSceneChangeThreshold(DEFAULT_SCENE_CHANGE_THRESHOLD),
	m_iBufferFrames(DEFAULT_BUFFER_FRAMES),
	m_iTotalFrameDelta(0),
	m_iTotalFrameDelta2(0),
	m_iSearchRadius(0),
	m_iIntActiveState(Active),
	m_dSourceFPS(0.0),
	m_dTargetFPS(0.0),
    m_iFrameRateMode(FRDisplayRate),
	m_dOFCCalcTime(0.0),
	m_dWarpCalcTime(0.0),
	m_bIsInitialized(false),
	m_pSettingsInterface(nullptr),
	m_hBrushGreen(CreateSolidBrush(GetSysColor(COLOR_BTNFACE))),
	m_hBrushOrange(CreateSolidBrush(GetSysColor(COLOR_BTNFACE))),
	m_hBrushRed(CreateSolidBrush(GetSysColor(COLOR_BTNFACE))) {
	ASSERT(phr);
}

// Destructor
CHopperRenderSettings::~CHopperRenderSettings() {
	if (m_hBrushGreen) DeleteObject(m_hBrushGreen);
	if (m_hBrushOrange) DeleteObject(m_hBrushOrange);
	if (m_hBrushRed) DeleteObject(m_hBrushRed);
}

// Handles the messages for our property window
INT_PTR CHopperRenderSettings::OnReceiveMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	switch (uMsg) {
		case WM_CTLCOLORSTATIC: {
			HDC hdcStatic = (HDC)wParam;
			HWND hwndStatic = (HWND)lParam;
			int ctrlId = GetDlgCtrlID(hwndStatic);

			if (ctrlId == IDC_TOTALFRAMEDELTA || ctrlId == IDC_TOTALFRAMEDELTA2) {
				bool delta1AboveThreshold = m_iTotalFrameDelta >= static_cast<unsigned int>(m_iSceneChangeThreshold);
				bool delta2AboveThreshold = m_iTotalFrameDelta2 >= static_cast<unsigned int>(m_iSceneChangeThreshold);

				COLORREF textColor;
				if (delta1AboveThreshold && delta2AboveThreshold) {
					// Both above threshold - Red
					textColor = RGB(220, 50, 50);
				} else if ((ctrlId == IDC_TOTALFRAMEDELTA && delta1AboveThreshold) ||
				           (ctrlId == IDC_TOTALFRAMEDELTA2 && delta2AboveThreshold)) {
					// Only this one above threshold - Orange
					textColor = RGB(230, 150, 0);
				} else {
					// Below threshold - Green
					textColor = RGB(50, 180, 50);
				}

				SetTextColor(hdcStatic, textColor);
				SetBkMode(hdcStatic, TRANSPARENT);
				return (INT_PTR)GetSysColorBrush(COLOR_BTNFACE);
			}
			break;
		}
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
					m_iFrameRateMode = FRDisplayRate; // Uses the display refresh rate
					m_dTargetFPS = 0.0;
					m_iDeltaScalar = DEFAULT_DELTA_SCALAR;
					m_iNeighborScalar = DEFAULT_NEIGHBOR_SCALAR;
					m_iBlackLevel = DEFAULT_BLACK_LEVEL;
					m_iWhiteLevel = DEFAULT_WHITE_LEVEL;
					m_iSceneChangeThreshold = DEFAULT_SCENE_CHANGE_THRESHOLD;
					m_iBufferFrames = DEFAULT_BUFFER_FRAMES;
					CheckRadioButton(m_Dlg, IDC_ON, IDC_OFF, IDC_ON); 
					CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLENDEDFRAMENOWARP, IDC_BLENDEDFRAME);
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
					(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iSceneChangeThreshold);
					Edit_SetText(GetDlgItem(m_Dlg, IDC_SCENECHANGETHRESHOLD), sz);
					(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%u\0"), m_iBufferFrames);
					Edit_SetText(GetDlgItem(m_Dlg, IDC_BUFFERFRAMES), sz);
					CheckDlgButton(m_Dlg, IDC_DEFAULTS, BST_UNCHECKED);
					SelectFrameRateMode(m_iFrameRateMode);
					UpdateTargetFPSFieldState();
				}
				// Keep the Target FPS field enabled state in sync with the frame rate mode
				if (action == CBN_SELCHANGE && nID == IDC_FRAMERATEMODE) {
					UpdateTargetFPSFieldState();
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
			m_pSettingsInterface->GetCurrentSettings(&m_bActivated, &frameOutput, &m_dTargetFPS, &m_iFrameRateMode, &m_iDeltaScalar, &m_iNeighborScalar, &m_iBlackLevel, &m_iWhiteLevel, &m_iSceneChangeThreshold,
									 &activeState, &m_dSourceFPS, &m_dOFCCalcTime, &m_dAVGOFCCalcTime, &m_dPeakOFCCalcTime, &m_dWarpCalcTime, &iDimX, &iDimY, &iLowDimX, &iLowDimY, &m_iTotalFrameDelta, &m_iTotalFrameDelta2, &m_iBufferFrames, &m_iSearchRadius);
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

			// Update the search radius
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iSearchRadius);
			SetDlgItemText(m_Dlg, IDC_SEARCHRADIUS, sz);

			// Update the total frame delta (peak scene change delta1)
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%u\0"), m_iTotalFrameDelta);
			SetDlgItemText(m_Dlg, IDC_TOTALFRAMEDELTA, sz);

			// Update the corresponding scene change delta2 at the peak
			(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%u\0"), m_iTotalFrameDelta2);
			SetDlgItemText(m_Dlg, IDC_TOTALFRAMEDELTA2, sz);

			// Refresh the Target FPS field with the effective rate, except in Custom mode where the user edits it
			if (m_bIsInitialized) {
				HWND hCombo = GetDlgItem(m_Dlg, IDC_FRAMERATEMODE);
				int sel = ComboBox_GetCurSel(hCombo);
				int selectedMode = (sel != CB_ERR) ? static_cast<int>(ComboBox_GetItemData(hCombo, sel)) : FRDisplayRate;
				if (selectedMode != FRCustom) {
					(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%.3f\0"), m_dTargetFPS);
					SetDlgItemText(m_Dlg, IDC_TARGETFPS, sz);
				}
			}
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
	m_pSettingsInterface->GetCurrentSettings(&m_bActivated, &frameOutput, &m_dTargetFPS, &m_iFrameRateMode, &m_iDeltaScalar, &m_iNeighborScalar, &m_iBlackLevel, &m_iWhiteLevel, &m_iSceneChangeThreshold,
	                                         &activeState, &m_dSourceFPS, &m_dOFCCalcTime, &m_dAVGOFCCalcTime, &m_dPeakOFCCalcTime, &m_dWarpCalcTime, &iDimX, &iDimY, &iLowDimX, &iLowDimY, &m_iTotalFrameDelta, &m_iTotalFrameDelta2, &m_iBufferFrames, &m_iSearchRadius);
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
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLENDEDFRAMENOWARP, IDC_WARPEDFRAME12);
			break;
		case WarpedFrame21:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLENDEDFRAMENOWARP, IDC_WARPEDFRAME21);
			break;
		case BlendedFrame:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLENDEDFRAMENOWARP, IDC_BLENDEDFRAME);
			break;
		case BlendedFrameNoWarp:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLENDEDFRAMENOWARP, IDC_BLENDEDFRAMENOWARP);
			break;
		case HSVFlow:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLENDEDFRAMENOWARP, IDC_HSVFLOW);
			break;
		case GreyFlow:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLENDEDFRAMENOWARP, IDC_GREYFLOW);
			break;
		case SideBySide1:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLENDEDFRAMENOWARP, IDC_SIDEBYSIDE1);
			break;
		case SideBySide2:
			CheckRadioButton(m_Dlg, IDC_WARPEDFRAME12, IDC_BLENDEDFRAMENOWARP, IDC_SIDEBYSIDE2);
			break;
	}

	// Set up the frame rate mode combo box
	PopulateFrameRateModeCombo();
	SelectFrameRateMode(m_iFrameRateMode);
	UpdateTargetFPSFieldState();

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

	// Set the initial Scene Change Threshold
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), m_iSceneChangeThreshold);
	Edit_SetText(GetDlgItem(m_Dlg, IDC_SCENECHANGETHRESHOLD), sz);

	// Set the initial Buffer Frames
	(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%u\0"), m_iBufferFrames);
	Edit_SetText(GetDlgItem(m_Dlg, IDC_BUFFERFRAMES), sz);

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
	ValidateParameter(m_iSceneChangeThreshold, 100000, IDC_SCENECHANGETHRESHOLD);
	// Validate buffer frames (must be positive, max 10)
	int bufferFramesInt = static_cast<int>(m_iBufferFrames);
	ValidateParameter(bufferFramesInt, 1000, IDC_BUFFERFRAMES);
	m_iBufferFrames = static_cast<unsigned int>(bufferFramesInt);

	// Tell the filter about the new settings
	m_pSettingsInterface->UpdateUserSettings(m_bActivated, m_iFrameOutput, m_dTargetFPS, m_iFrameRateMode, m_iDeltaScalar, m_iNeighborScalar, m_iBlackLevel, m_iWhiteLevel, m_iSceneChangeThreshold, m_iBufferFrames);

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
	unsigned int totalFrameDelta;
	unsigned int totalFrameDelta2;
	unsigned int bufferFrames;
	int searchRadius;
	CheckPointer(m_pSettingsInterface, E_FAIL);
	m_pSettingsInterface->GetCurrentSettings(&m_bActivated, &frameOutput, &m_dTargetFPS, &m_iFrameRateMode, &m_iDeltaScalar, &m_iNeighborScalar, &m_iBlackLevel, &m_iWhiteLevel,
	                                         &m_iSceneChangeThreshold, &activeState, &m_dSourceFPS, &m_dOFCCalcTime, &m_dAVGOFCCalcTime, &m_dPeakOFCCalcTime, &m_dWarpCalcTime, &iDimX, &iDimY, &iLowDimX, &iLowDimY, &totalFrameDelta, &totalFrameDelta2, &bufferFrames, &searchRadius);
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
	} else if (IsDlgButtonChecked(m_Dlg, IDC_BLENDEDFRAMENOWARP)) {
		m_iFrameOutput = BlendedFrameNoWarp;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_HSVFLOW)) {
		m_iFrameOutput = HSVFlow;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_GREYFLOW)) {
		m_iFrameOutput = GreyFlow;
	} else if (IsDlgButtonChecked(m_Dlg, IDC_SIDEBYSIDE1)) {
		m_iFrameOutput = SideBySide1;
	} else {
		m_iFrameOutput = SideBySide2;
	}

	// Get the selected frame rate mode from the combo box
	HWND hCombo = GetDlgItem(m_Dlg, IDC_FRAMERATEMODE);
	int sel = ComboBox_GetCurSel(hCombo);
	if (sel != CB_ERR) {
		m_iFrameRateMode = static_cast<int>(ComboBox_GetItemData(hCombo, sel));
	}

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

	// Get the scene change threshold
	Edit_GetText(GetDlgItem(m_Dlg, IDC_SCENECHANGETHRESHOLD), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
	int tmp6 = atoi(szANSI);
#else
    int tmp6 = atoi(sz);
#endif

	// Get the buffer frames
	Edit_GetText(GetDlgItem(m_Dlg, IDC_BUFFERFRAMES), sz, STR_MAX_LENGTH);

#ifdef UNICODE
	// Convert Multibyte string to ANSI
	rc = WideCharToMultiByte(CP_ACP, 0, sz, -1, szANSI, STR_MAX_LENGTH, nullptr, nullptr);
	unsigned int tmp7 = static_cast<unsigned int>(atoi(szANSI));
#else
    unsigned int tmp7 = static_cast<unsigned int>(atoi(sz));
#endif

	m_dTargetFPS = tmp1;
	m_iDeltaScalar = tmp2;
	m_iNeighborScalar = tmp3;
	m_iBlackLevel = tmp4;
	m_iWhiteLevel = tmp5;
	m_iSceneChangeThreshold = tmp6;
	m_iBufferFrames = tmp7;
}

// Populates the frame rate mode combo box with the available target frame rate options
void CHopperRenderSettings::PopulateFrameRateModeCombo() {
	HWND hCombo = GetDlgItem(m_Dlg, IDC_FRAMERATEMODE);
	if (!hCombo) {
		return;
	}
	ComboBox_ResetContent(hCombo);

	int idx = ComboBox_AddString(hCombo, TEXT("Display Rate"));
	ComboBox_SetItemData(hCombo, idx, FRDisplayRate);
	idx = ComboBox_AddString(hCombo, TEXT("1/2x Display Rate"));
	ComboBox_SetItemData(hCombo, idx, FRHalfDisplayRate);
	idx = ComboBox_AddString(hCombo, TEXT("Custom"));
	ComboBox_SetItemData(hCombo, idx, FRCustom);

	const int multipliers[] = { 2, 3, 4, 5, 6, 8 };
	for (int multiplier : multipliers) {
		TCHAR sz[16];
		(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%dx Source"), multiplier);
		idx = ComboBox_AddString(hCombo, sz);
		ComboBox_SetItemData(hCombo, idx, multiplier);
	}
}

// Selects the combo box entry that matches the given frame rate mode
void CHopperRenderSettings::SelectFrameRateMode(int mode) {
	HWND hCombo = GetDlgItem(m_Dlg, IDC_FRAMERATEMODE);
	if (!hCombo) {
		return;
	}

	int count = ComboBox_GetCount(hCombo);
	for (int i = 0; i < count; i++) {
		if (static_cast<int>(ComboBox_GetItemData(hCombo, i)) == mode) {
			ComboBox_SetCurSel(hCombo, i);
			return;
		}
	}

	// Unknown multiplier not in the predefined list: add and select it
	if (mode >= 2) {
		TCHAR sz[16];
		(void)StringCchPrintf(sz, NUMELMS(sz), TEXT("%dx Source"), mode);
		int idx = ComboBox_AddString(hCombo, sz);
		ComboBox_SetItemData(hCombo, idx, mode);
		ComboBox_SetCurSel(hCombo, idx);
		return;
	}

	// Fall back to the first entry (Display Rate)
	ComboBox_SetCurSel(hCombo, 0);
}

// Enables the Target FPS edit field only when the Custom frame rate mode is selected
void CHopperRenderSettings::UpdateTargetFPSFieldState() {
	HWND hCombo = GetDlgItem(m_Dlg, IDC_FRAMERATEMODE);
	int mode = FRDisplayRate;
	if (hCombo) {
		int sel = ComboBox_GetCurSel(hCombo);
		if (sel != CB_ERR) {
			mode = static_cast<int>(ComboBox_GetItemData(hCombo, sel));
		}
	}
	EnableWindow(GetDlgItem(m_Dlg, IDC_TARGETFPS), mode == FRCustom);
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

		// Save the frame rate mode
		DWORD frameRateMode = static_cast<DWORD>(m_iFrameRateMode);
		LONG result4 = RegSetValueEx(hKey, L"FrameRateMode", 0, REG_DWORD, reinterpret_cast<BYTE*>(&frameRateMode), sizeof(DWORD));

		// Save the delta scalar
		LONG result5 = RegSetValueEx(hKey, L"DeltaScalar", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iDeltaScalar), sizeof(DWORD));

		// Save the neighbor scalar
		LONG result6 = RegSetValueEx(hKey, L"NeighborScalar", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iNeighborScalar), sizeof(DWORD));

		// Save the black level
		LONG result7 = RegSetValueEx(hKey, L"BlackLevel", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iBlackLevel), sizeof(DWORD));

		// Save the white level
		LONG result8 = RegSetValueEx(hKey, L"WhiteLevel", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iWhiteLevel), sizeof(DWORD));

		// Save the scene change threshold
		LONG result9 = RegSetValueEx(hKey, L"SceneChangeThreshold", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iSceneChangeThreshold), sizeof(DWORD));

		// Save the buffer frames
		LONG result10 = RegSetValueEx(hKey, L"BufferFrames", 0, REG_DWORD, reinterpret_cast<BYTE*>(&m_iBufferFrames), sizeof(DWORD));
		
		RegCloseKey(hKey); // Close the registry key

		// Check for errors
        if (result1 || result2 || result3 || result4 || result5 || result6 || result7 || result8 || result9 || result10) {
			return E_FAIL;
        } else {
            return S_OK;
        }

    } else {
        return E_FAIL;
    }
}
