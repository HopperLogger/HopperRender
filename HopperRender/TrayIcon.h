#pragma once

#include <windows.h>
#include <shellapi.h>

class CHopperRender;

// System tray icon for HopperRender
class CTrayIcon {
public:
    explicit CTrayIcon(CHopperRender* pFilter);
    ~CTrayIcon();

    CTrayIcon(const CTrayIcon&) = delete;
    CTrayIcon& operator=(const CTrayIcon&) = delete;

private:
    static DWORD WINAPI ThreadProc(LPVOID lpParam);
    static LRESULT CALLBACK WndProcStatic(HWND, UINT, WPARAM, LPARAM);
    LRESULT WndProc(HWND, UINT, WPARAM, LPARAM);

    void Run();
    void ShowContextMenu(HWND hWnd);
    void BuildMenu();
    void RefreshMenuItems();
    void UpdateIconState(bool force);
    HICON LoadPngIconFromResource(int resourceId);

    static void SaveDwordToRegistry(LPCWSTR valueName, DWORD value);

    CHopperRender* m_pFilter;
    HANDLE m_hThread;
    DWORD  m_dwThreadId;
    HWND   m_hWnd;
    HICON  m_hIconActive;
    HICON  m_hIconInactive;
    HMENU  m_hMenu;
    HMENU  m_hMenuFrameOut;
    bool   m_bIconIsActive;
    bool   m_bMenuVisible;
    ULONG_PTR m_gdiplusToken;
    NOTIFYICONDATAW m_nid;
    HANDLE m_hReadyEvent;
    double m_dLastTargetFPS;
};
