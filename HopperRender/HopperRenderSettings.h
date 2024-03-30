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
	int m_iNumIterations; // Number of iterations to use in the optical flow calculation (0: As many as possible)
	int m_iFrameBlurKernelSize; // The size of the blur kernel used to blur the source frames before calculating the optical flow
	int m_iFlowBlurKernelSize; // The size of the blur kernel used to blur the offset calculated by the optical flow
	int m_iSceneChangeThreshold; // The threshold used to determine whether a scene change has occurred
	ActiveState m_iIntActiveState; // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)
	double m_dSourceFPS; // The source frames per second
	int m_iNumSteps; // Number of steps executed to find the ideal offset (limits the maximum offset distance per iteration)
	int m_iCurrentSceneChange; // How many pixel differences are currently detected
	bool m_bIsInitialized; // Used to ignore startup messages
	unsigned char m_cRefreshCounter; // Counts the number of times the settings page has been refreshed
	SettingsInterface* m_pSettingsInterface; // The custom interface on the filter
};