#include <windows.h>
#include <streams.h>
#include <shellapi.h>
#include <shlwapi.h>
#include <objidl.h>
#include <gdiplus.h>
#include <strsafe.h>

#include "TrayIcon.h"
#include "HopperRender.h"
#include "resource.h"

#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "shlwapi.lib")

extern HINSTANCE g_hInst;

namespace {
    constexpr UINT WM_TRAY_CALLBACK     = WM_APP + 1;
    constexpr UINT TRAY_UID             = 1;
    constexpr UINT_PTR TIMER_POLL_STATE = 1; // background polling for icon swap
    constexpr UINT_PTR TIMER_MENU_REFRESH = 2; // refreshes stats while menu open

    constexpr UINT ID_TRAY_TOGGLE          = 30000;
    constexpr UINT ID_TRAY_FRAMEOUT_BASE   = 30100; // + FrameOutput enum value
    constexpr UINT ID_TRAY_STAT_BASE       = 30200; // disabled stats items

    constexpr int STAT_STATUS         = 0;
    constexpr int STAT_SOURCE_FPS     = 1;
    constexpr int STAT_TARGET_FPS     = 2;
    constexpr int STAT_OFC_TIME       = 3;
    constexpr int STAT_AVG_OFC_TIME   = 4;
    constexpr int STAT_PEAK_OFC_TIME  = 5;
    constexpr int STAT_WARP_TIME      = 6;
    constexpr int STAT_FRAME_RES      = 7;
    constexpr int STAT_CALC_RES       = 8;
    constexpr int STAT_SEARCH_RADIUS  = 9;
    constexpr int STAT_COUNT          = 10;

    const wchar_t* kFrameOutputNames[] = {
        L"WarpedFrame 1 \u2192 2",
        L"WarpedFrame 2 \u2192 1",
        L"BlendedFrame (Default)",
        L"HSV Flow",
        L"Grey Flow",
        L"Side-by-side 1",
        L"Side-by-side 2",
    };
    constexpr int kFrameOutputCount = sizeof(kFrameOutputNames) / sizeof(kFrameOutputNames[0]);

    // Display order shown in the tray submenu
    constexpr int kFrameOutputDisplayOrder[] = { 2, 0, 1, 3, 4, 5, 6 };

    const wchar_t kWindowClassName[] = L"HopperRenderTrayWnd";
}

CTrayIcon::CTrayIcon(CHopperRender* pFilter)
    : m_pFilter(pFilter)
    , m_hThread(nullptr)
    , m_dwThreadId(0)
    , m_hWnd(nullptr)
    , m_hIconActive(nullptr)
    , m_hIconInactive(nullptr)
    , m_hMenu(nullptr)
    , m_hMenuFrameOut(nullptr)
    , m_bIconIsActive(false)
    , m_bMenuVisible(false)
    , m_gdiplusToken(0)
    , m_nid{}
    , m_hReadyEvent(nullptr)
    , m_dLastTargetFPS(0.0)
{
    m_hReadyEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    m_hThread = CreateThread(nullptr, 0, &CTrayIcon::ThreadProc, this, 0, &m_dwThreadId);
    if (m_hThread && m_hReadyEvent) {
        // Wait briefly for the thread to set up the window so that destructor
        // can safely PostThreadMessage / PostMessage to it.
        WaitForSingleObject(m_hReadyEvent, 5000);
    }
}

CTrayIcon::~CTrayIcon() {
    if (m_hThread) {
        if (m_hWnd) {
            PostMessageW(m_hWnd, WM_CLOSE, 0, 0);
        } else if (m_dwThreadId) {
            PostThreadMessageW(m_dwThreadId, WM_QUIT, 0, 0);
        }
        WaitForSingleObject(m_hThread, 5000);
        CloseHandle(m_hThread);
        m_hThread = nullptr;
    }
    if (m_hReadyEvent) {
        CloseHandle(m_hReadyEvent);
        m_hReadyEvent = nullptr;
    }
}

DWORD WINAPI CTrayIcon::ThreadProc(LPVOID lpParam) {
    CTrayIcon* self = static_cast<CTrayIcon*>(lpParam);
    self->Run();
    return 0;
}

void CTrayIcon::Run() {
    Gdiplus::GdiplusStartupInput gpsi;
    Gdiplus::GdiplusStartup(&m_gdiplusToken, &gpsi, nullptr);

    m_hIconActive   = LoadPngIconFromResource(IDR_TRAY_ICON_ACTIVE);
    m_hIconInactive = LoadPngIconFromResource(IDR_TRAY_ICON_INACTIVE);
    if (!m_hIconActive)   m_hIconActive   = LoadIconW(nullptr, IDI_APPLICATION);
    if (!m_hIconInactive) m_hIconInactive = LoadIconW(nullptr, IDI_APPLICATION);

    WNDCLASSEXW wcex{};
    wcex.cbSize        = sizeof(wcex);
    wcex.lpfnWndProc   = &CTrayIcon::WndProcStatic;
    wcex.hInstance     = g_hInst;
    wcex.lpszClassName = kWindowClassName;
    RegisterClassExW(&wcex);

    m_hWnd = CreateWindowExW(0, kWindowClassName, L"HopperRenderTray",
                             0, 0, 0, 0, 0,
                             HWND_MESSAGE, nullptr, g_hInst, this);

    if (!m_hWnd) {
        if (m_hReadyEvent) SetEvent(m_hReadyEvent);
        Gdiplus::GdiplusShutdown(m_gdiplusToken);
        return;
    }

    SetWindowLongPtrW(m_hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

    BuildMenu();

    // Add the tray icon
    m_nid = {};
    m_nid.cbSize = sizeof(m_nid);
    m_nid.hWnd   = m_hWnd;
    m_nid.uID    = TRAY_UID;
    m_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    m_nid.uCallbackMessage = WM_TRAY_CALLBACK;
    m_nid.hIcon  = m_hIconInactive;
    StringCchCopyW(m_nid.szTip, ARRAYSIZE(m_nid.szTip), L"HopperRender");
    Shell_NotifyIconW(NIM_ADD, &m_nid);
    m_bIconIsActive = false;

    // Poll filter state at ~24 Hz to swap the icon if the active state changes
    SetTimer(m_hWnd, TIMER_POLL_STATE, 42, nullptr);

    UpdateIconState(true);

    if (m_hReadyEvent) SetEvent(m_hReadyEvent);

    // Message loop
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    // Cleanup
    Shell_NotifyIconW(NIM_DELETE, &m_nid);
    if (m_hMenu)        { DestroyMenu(m_hMenu); m_hMenu = nullptr; }
    if (m_hIconActive)   { DestroyIcon(m_hIconActive);   m_hIconActive = nullptr; }
    if (m_hIconInactive) { DestroyIcon(m_hIconInactive); m_hIconInactive = nullptr; }
    UnregisterClassW(kWindowClassName, g_hInst);

    Gdiplus::GdiplusShutdown(m_gdiplusToken);
    m_gdiplusToken = 0;
}

LRESULT CALLBACK CTrayIcon::WndProcStatic(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    CTrayIcon* self = reinterpret_cast<CTrayIcon*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
    if (self) {
        return self->WndProc(hWnd, uMsg, wParam, lParam);
    }
    return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

LRESULT CTrayIcon::WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_TRAY_CALLBACK: {
        UINT mouseMsg = LOWORD(lParam);
        if (mouseMsg == WM_RBUTTONUP || mouseMsg == WM_LBUTTONUP || mouseMsg == WM_CONTEXTMENU) {
            ShowContextMenu(hWnd);
        }
        return 0;
    }
    case WM_TIMER:
        if (wParam == TIMER_POLL_STATE) {
            UpdateIconState(false);
        } else if (wParam == TIMER_MENU_REFRESH) {
            if (m_bMenuVisible) {
                RefreshMenuItems();
            }
        }
        return 0;
    case WM_COMMAND: {
        const UINT id = LOWORD(wParam);
        if (id == ID_TRAY_TOGGLE) {
            // Read current settings, flip activated, push back
            bool bActivated; int iFrameOutput; double dTargetFPS; int iFrameRateMode;
            int iDeltaScalar; int iNeighborScalar; int iBlackLevel; int iWhiteLevel;
            int iSceneChangeThreshold; int iIntActiveState;
            double dSourceFPS, dOFCCalcTime, dAVGOFCCalcTime, dPeakOFCCalcTime, dWarpCalcTime;
            int iDimX, iDimY, iLowDimX, iLowDimY, iSearchRadius;
            unsigned int iTotalFrameDelta, iTotalFrameDelta2, iBufferFrames;
            m_pFilter->GetCurrentSettings(&bActivated, &iFrameOutput, &dTargetFPS, &iFrameRateMode,
                &iDeltaScalar, &iNeighborScalar, &iBlackLevel, &iWhiteLevel,
                &iSceneChangeThreshold, &iIntActiveState,
                &dSourceFPS, &dOFCCalcTime, &dAVGOFCCalcTime, &dPeakOFCCalcTime, &dWarpCalcTime,
                &iDimX, &iDimY, &iLowDimX, &iLowDimY,
                &iTotalFrameDelta, &iTotalFrameDelta2, &iBufferFrames, &iSearchRadius);

            bActivated = !bActivated;
            m_pFilter->UpdateUserSettings(bActivated, iFrameOutput, dTargetFPS, iFrameRateMode,
                iDeltaScalar, iNeighborScalar, iBlackLevel, iWhiteLevel,
                iSceneChangeThreshold, iBufferFrames);
            SaveDwordToRegistry(L"Activated", bActivated ? 1u : 0u);
            UpdateIconState(true);
        } else if (id >= ID_TRAY_FRAMEOUT_BASE && id < ID_TRAY_FRAMEOUT_BASE + kFrameOutputCount) {
            const int newFrameOutput = static_cast<int>(id - ID_TRAY_FRAMEOUT_BASE);
            bool bActivated; int iFrameOutput; double dTargetFPS; int iFrameRateMode;
            int iDeltaScalar; int iNeighborScalar; int iBlackLevel; int iWhiteLevel;
            int iSceneChangeThreshold; int iIntActiveState;
            double dSourceFPS, dOFCCalcTime, dAVGOFCCalcTime, dPeakOFCCalcTime, dWarpCalcTime;
            int iDimX, iDimY, iLowDimX, iLowDimY, iSearchRadius;
            unsigned int iTotalFrameDelta, iTotalFrameDelta2, iBufferFrames;
            m_pFilter->GetCurrentSettings(&bActivated, &iFrameOutput, &dTargetFPS, &iFrameRateMode,
                &iDeltaScalar, &iNeighborScalar, &iBlackLevel, &iWhiteLevel,
                &iSceneChangeThreshold, &iIntActiveState,
                &dSourceFPS, &dOFCCalcTime, &dAVGOFCCalcTime, &dPeakOFCCalcTime, &dWarpCalcTime,
                &iDimX, &iDimY, &iLowDimX, &iLowDimY,
                &iTotalFrameDelta, &iTotalFrameDelta2, &iBufferFrames, &iSearchRadius);

            m_pFilter->UpdateUserSettings(bActivated, newFrameOutput, dTargetFPS, iFrameRateMode,
                iDeltaScalar, iNeighborScalar, iBlackLevel, iWhiteLevel,
                iSceneChangeThreshold, iBufferFrames);
            SaveDwordToRegistry(L"FrameOutput", static_cast<DWORD>(newFrameOutput));
        }
        return 0;
    }
    case WM_CLOSE:
        DestroyWindow(hWnd);
        return 0;
    case WM_DESTROY:
        KillTimer(hWnd, TIMER_POLL_STATE);
        PostQuitMessage(0);
        return 0;
    default:
        // Re-add icon if Explorer restarted
        static UINT s_taskbarCreated = RegisterWindowMessageW(L"TaskbarCreated");
        if (uMsg == s_taskbarCreated) {
            Shell_NotifyIconW(NIM_ADD, &m_nid);
            UpdateIconState(true);
            return 0;
        }
        break;
    }
    return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

void CTrayIcon::BuildMenu() {
    m_hMenu = CreatePopupMenu();
    m_hMenuFrameOut = CreatePopupMenu();

    for (int i = 0; i < kFrameOutputCount; ++i) {
        const int enumVal = kFrameOutputDisplayOrder[i];
        AppendMenuW(m_hMenuFrameOut, MF_STRING, ID_TRAY_FRAMEOUT_BASE + enumVal, kFrameOutputNames[enumVal]);
    }

    AppendMenuW(m_hMenu, MF_STRING, ID_TRAY_TOGGLE, L"Enable Interpolation");
    AppendMenuW(m_hMenu, MF_POPUP, reinterpret_cast<UINT_PTR>(m_hMenuFrameOut), L"Frame Output");
    AppendMenuW(m_hMenu, MF_SEPARATOR, 0, nullptr);

    // Placeholder stat entries; populated when shown
    for (int i = 0; i < STAT_COUNT; ++i) {
        AppendMenuW(m_hMenu, MF_STRING | MF_GRAYED | MF_DISABLED, ID_TRAY_STAT_BASE + i, L"");
    }
}

void CTrayIcon::ShowContextMenu(HWND hWnd) {
    POINT pt;
    GetCursorPos(&pt);

    RefreshMenuItems();

    // Foreground required for menu to dismiss on outside click (MSDN trick)
    SetForegroundWindow(hWnd);

    m_bMenuVisible = true;
    SetTimer(hWnd, TIMER_MENU_REFRESH, 42, nullptr);

    TrackPopupMenu(m_hMenu, TPM_BOTTOMALIGN | TPM_LEFTALIGN | TPM_RIGHTBUTTON,
                   pt.x, pt.y, 0, hWnd, nullptr);

    KillTimer(hWnd, TIMER_MENU_REFRESH);
    m_bMenuVisible = false;

    PostMessageW(hWnd, WM_NULL, 0, 0);
}

void CTrayIcon::RefreshMenuItems() {
    if (!m_hMenu || !m_pFilter) return;

    bool bActivated; int iFrameOutput; int iFrameRateMode;
    // Init to 0.0 so the filter reports the effective target FPS
    double dTargetFPS = 0.0;
    int iDeltaScalar; int iNeighborScalar; int iBlackLevel; int iWhiteLevel;
    int iSceneChangeThreshold; int iIntActiveState;
    double dSourceFPS, dOFCCalcTime, dAVGOFCCalcTime, dPeakOFCCalcTime, dWarpCalcTime;
    int iDimX, iDimY, iLowDimX, iLowDimY, iSearchRadius;
    unsigned int iTotalFrameDelta, iTotalFrameDelta2, iBufferFrames;
    m_pFilter->GetCurrentSettings(&bActivated, &iFrameOutput, &dTargetFPS, &iFrameRateMode,
        &iDeltaScalar, &iNeighborScalar, &iBlackLevel, &iWhiteLevel,
        &iSceneChangeThreshold, &iIntActiveState,
        &dSourceFPS, &dOFCCalcTime, &dAVGOFCCalcTime, &dPeakOFCCalcTime, &dWarpCalcTime,
        &iDimX, &iDimY, &iLowDimX, &iLowDimY,
        &iTotalFrameDelta, &iTotalFrameDelta2, &iBufferFrames, &iSearchRadius);

    // Toggle check state
    CheckMenuItem(m_hMenu, ID_TRAY_TOGGLE,
                  MF_BYCOMMAND | (bActivated ? MF_CHECKED : MF_UNCHECKED));

    // Frame output radio check
    if (iFrameOutput >= 0 && iFrameOutput < kFrameOutputCount) {
        CheckMenuRadioItem(m_hMenuFrameOut,
                           ID_TRAY_FRAMEOUT_BASE,
                           ID_TRAY_FRAMEOUT_BASE + kFrameOutputCount - 1,
                           ID_TRAY_FRAMEOUT_BASE + iFrameOutput,
                           MF_BYCOMMAND);
    }

    const wchar_t* stateStr = L"Unknown";
    switch (iIntActiveState) {
        case Deactivated: stateStr = L"Deactivated"; break;
        case NotNeeded:   stateStr = L"Not Needed"; break;
        case Active:      stateStr = L"Active"; break;
        case TooSlow:     stateStr = L"Too Slow"; break;
    }

    wchar_t buf[128];

    auto setStat = [&](int slot, const wchar_t* text) {
        MENUITEMINFOW mii{};
        mii.cbSize = sizeof(mii);
        mii.fMask = MIIM_STRING;
        mii.dwTypeData = const_cast<LPWSTR>(text);
        SetMenuItemInfoW(m_hMenu, ID_TRAY_STAT_BASE + slot, FALSE, &mii);
    };

    StringCchPrintfW(buf, ARRAYSIZE(buf), L"Status: %s", stateStr);
    setStat(STAT_STATUS, buf);
    StringCchPrintfW(buf, ARRAYSIZE(buf), L"Source FPS: %.3f", dSourceFPS);
    setStat(STAT_SOURCE_FPS, buf);
    // Filter transiently reports 0 (or absurdly small values during state changes); cache last sensible value to avoid flicker
    if (dTargetFPS >= 1.0) {
        m_dLastTargetFPS = dTargetFPS;
    }
    const double displayTargetFPS = (m_dLastTargetFPS >= 1.0) ? m_dLastTargetFPS : dTargetFPS;
    const wchar_t* fpsModeSuffix = (iFrameRateMode == FRDisplayRate) ? L" (display)"
                                 : (iFrameRateMode == FRHalfDisplayRate) ? L" (1/2x display)"
                                 : (iFrameRateMode == FRCustom) ? L" (custom)"
                                 : L" (multiplier)";
    StringCchPrintfW(buf, ARRAYSIZE(buf), L"Target FPS: %.3f%s", displayTargetFPS, fpsModeSuffix);
    setStat(STAT_TARGET_FPS, buf);
    StringCchPrintfW(buf, ARRAYSIZE(buf), L"OFC Calc Time: %.2f ms", dOFCCalcTime);
    setStat(STAT_OFC_TIME, buf);
    StringCchPrintfW(buf, ARRAYSIZE(buf), L"Avg OFC Time: %.2f ms", dAVGOFCCalcTime);
    setStat(STAT_AVG_OFC_TIME, buf);
    StringCchPrintfW(buf, ARRAYSIZE(buf), L"Peak OFC Time: %.2f ms", dPeakOFCCalcTime);
    setStat(STAT_PEAK_OFC_TIME, buf);
    StringCchPrintfW(buf, ARRAYSIZE(buf), L"Warp Calc Time: %.2f ms", dWarpCalcTime);
    setStat(STAT_WARP_TIME, buf);
    StringCchPrintfW(buf, ARRAYSIZE(buf), L"Frame Resolution: %d x %d", iDimX, iDimY);
    setStat(STAT_FRAME_RES, buf);
    StringCchPrintfW(buf, ARRAYSIZE(buf), L"Calc Resolution: %d x %d", iLowDimX, iLowDimY);
    setStat(STAT_CALC_RES, buf);
    StringCchPrintfW(buf, ARRAYSIZE(buf), L"Search Radius: %d", iSearchRadius);
    setStat(STAT_SEARCH_RADIUS, buf);
}

void CTrayIcon::UpdateIconState(bool force) {
    if (!m_pFilter || !m_hWnd) return;

    bool bActivated; int iFrameOutput; double dTargetFPS; int iFrameRateMode;
    int iDeltaScalar; int iNeighborScalar; int iBlackLevel; int iWhiteLevel;
    int iSceneChangeThreshold; int iIntActiveState;
    double dSourceFPS, dOFCCalcTime, dAVGOFCCalcTime, dPeakOFCCalcTime, dWarpCalcTime;
    int iDimX, iDimY, iLowDimX, iLowDimY, iSearchRadius;
    unsigned int iTotalFrameDelta, iTotalFrameDelta2, iBufferFrames;
    m_pFilter->GetCurrentSettings(&bActivated, &iFrameOutput, &dTargetFPS, &iFrameRateMode,
        &iDeltaScalar, &iNeighborScalar, &iBlackLevel, &iWhiteLevel,
        &iSceneChangeThreshold, &iIntActiveState,
        &dSourceFPS, &dOFCCalcTime, &dAVGOFCCalcTime, &dPeakOFCCalcTime, &dWarpCalcTime,
        &iDimX, &iDimY, &iLowDimX, &iLowDimY,
        &iTotalFrameDelta, &iTotalFrameDelta2, &iBufferFrames, &iSearchRadius);

    const bool nowActive = (iIntActiveState == Active);
    if (force || nowActive != m_bIconIsActive) {
        m_bIconIsActive = nowActive;
        m_nid.uFlags = NIF_ICON | NIF_TIP;
        m_nid.hIcon = nowActive ? m_hIconActive : m_hIconInactive;
        StringCchCopyW(m_nid.szTip, ARRAYSIZE(m_nid.szTip),
                       nowActive ? L"HopperRender \u2014 Active"
                                 : L"HopperRender \u2014 Not interpolating");
        Shell_NotifyIconW(NIM_MODIFY, &m_nid);
    }
}

HICON CTrayIcon::LoadPngIconFromResource(int resourceId) {
    HRSRC hRes = FindResourceW(g_hInst, MAKEINTRESOURCEW(resourceId), RT_RCDATA);
    if (!hRes) return nullptr;
    const DWORD size = SizeofResource(g_hInst, hRes);
    HGLOBAL hMem = LoadResource(g_hInst, hRes);
    if (!hMem || size == 0) return nullptr;
    void* pData = LockResource(hMem);
    if (!pData) return nullptr;

    IStream* pStream = SHCreateMemStream(static_cast<const BYTE*>(pData), size);
    if (!pStream) return nullptr;

    HICON hIcon = nullptr;
    Gdiplus::Bitmap* bmp = Gdiplus::Bitmap::FromStream(pStream);
    if (bmp && bmp->GetLastStatus() == Gdiplus::Ok) {
        bmp->GetHICON(&hIcon);
    }
    delete bmp;
    pStream->Release();
    return hIcon;
}

void CTrayIcon::SaveDwordToRegistry(LPCWSTR valueName, DWORD value) {
    HKEY hKey;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, L"SOFTWARE\\HopperRender", 0, nullptr,
                        REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, nullptr, &hKey, nullptr) == ERROR_SUCCESS) {
        RegSetValueExW(hKey, valueName, 0, REG_DWORD,
                       reinterpret_cast<const BYTE*>(&value), sizeof(DWORD));
        RegCloseKey(hKey);
    }
}
