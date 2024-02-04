#include <strsafe.h>

class CHopperRenderSettings : public CBasePropertyPage
{

public:

    static CUnknown * WINAPI CreateInstance(LPUNKNOWN lpunk, HRESULT *phr);

private:

    INT_PTR OnReceiveMessage(HWND hwnd,UINT uMsg,WPARAM wParam,LPARAM lParam);
    HRESULT OnConnect(IUnknown *pUnknown);
    HRESULT OnDisconnect();
    HRESULT OnActivate();
    HRESULT OnDeactivate();
    HRESULT OnApplyChanges();

    void    GetControlValues();

    CHopperRenderSettings(LPUNKNOWN lpunk, HRESULT *phr);

    BOOL m_bIsInitialized;      // Used to ignore startup messages
    int m_effect;               // Which effect are we processing
    int m_numSteps;             // Number of steps executed to find the ideal offset (limits the maximum offset)
    int m_maxOffsetDivider;     // The divider used to calculate the initial global offset
    IIPEffect *m_pIPEffect;     // The custom interface on the filter

}; // HopperRenderSettings

