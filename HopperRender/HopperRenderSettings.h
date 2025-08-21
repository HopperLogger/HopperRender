#pragma once

#include <strsafe.h>
#include "HopperRender.h"

class CHopperRenderSettings : public CBasePropertyPage {
public:
	static CUnknown* WINAPI CreateInstance(LPUNKNOWN lpunk, HRESULT* phr);

private:
	INT_PTR OnReceiveMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) override;
	HRESULT OnConnect(IUnknown* pUnknown) override;
	HRESULT OnDisconnect() override;
	HRESULT OnActivate() override;
	HRESULT OnDeactivate() override;
	void ValidateParameter(int& parameter, int maxValue, int controlId);
	HRESULT OnApplyChanges() override;
	HRESULT saveSettings();

	void GetControlValues();

	CHopperRenderSettings(LPUNKNOWN lpunk, HRESULT* phr);

	bool m_bActivated; // Whether the filter is activated by the user
	FrameOutput m_iFrameOutput; // What frame output to use (0: WarpedFrame 1 -> 2, 1: WarpedFrame 2 -> 1, 2: BlendedFrame, 3: HSV Flow, 4: Blurred Frames, 5: Side-by-side 1, 6: Side-by-side 2)
	int m_iDeltaScalar;
	int m_iNeighborScalar;
	int m_iBlackLevel;
	int m_iWhiteLevel;
	ActiveState m_iIntActiveState; // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)
	double m_dSourceFPS; // The source frames per second
	double m_dTargetFPS;
	double m_dOFCCalcTime;
	double m_dWarpCalcTime;
	bool m_bIsInitialized; // Used to ignore startup messages
	SettingsInterface* m_pSettingsInterface; // The custom interface on the filter
};
