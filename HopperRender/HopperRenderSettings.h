#pragma once

#include <strsafe.h>

class CHopperRenderSettings : public CBasePropertyPage {
public:
	static CUnknown* WINAPI CreateInstance(LPUNKNOWN lpunk, HRESULT* phr);

private:
	INT_PTR OnReceiveMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) override;
	HRESULT OnConnect(IUnknown* pUnknown) override;
	HRESULT OnDisconnect() override;
	HRESULT OnActivate() override;
	HRESULT OnDeactivate() override;
	HRESULT OnApplyChanges() override;

	void GetControlValues();

	CHopperRenderSettings(LPUNKNOWN lpunk, HRESULT* phr);

	BOOL m_bIsInitialized; // Used to ignore startup messages
	bool m_bActivated; // Whether the filter is activated
	int m_iFrameOutput; // What frame output to use
	int m_iNumIterations; // Number of iterations to use in the optical flow calculation
	int m_iBlurKernelSize; // The size of the blur kernel
	int m_iIntActiveState; // The state of the filter
	double m_dSourceFPS; // The source frames per second
	int m_iNumSteps; // The number of steps to use in the optical flow calculation
	SettingsInterface* m_pSettingsInterface; // The custom interface on the filter
};