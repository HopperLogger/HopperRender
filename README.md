<div align="center">
  <img alt="HopperRender logo" height="200px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/faa253f0-3276-4404-aa4a-bb9c2b35056c">

  <h1>HopperRender</h1>

  <p><strong>Real-time optical flow frame interpolation for DirectShow players.</strong><br>
  Watch any local video at the native refresh rate of your monitor.</p>

  <p>
    <a href="https://github.com/HopperLogger/HopperRender/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/HopperLogger/HopperRender?style=flat-square"></a>
    <a href="https://github.com/HopperLogger/HopperRender/releases"><img alt="Downloads" src="https://img.shields.io/github/downloads/HopperLogger/HopperRender/total?style=flat-square"></a>
    <a href="LICENSE"><img alt="License: GPL-3.0" src="https://img.shields.io/github/license/HopperLogger/HopperRender?style=flat-square"></a>
    <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-0078D6?style=flat-square">
    <img alt="GPU: OpenCL" src="https://img.shields.io/badge/GPU-OpenCL%201.2%2B-ED1C24?style=flat-square">
  </p>

  <p>
    <a href="#installation">Installation</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#how-it-works">How it works</a> •
    <a href="#building-from-source">Building</a> •
    <a href="CHANGELOG.md">Changelog</a>
  </p>
</div>

---

HopperRender is a DirectShow video filter that generates intermediate frames using GPU-accelerated optical flow, interpolating any source frame rate up to your display's refresh rate. It plugs into DirectShow media players such as [MPC-HC](https://github.com/clsid2/mpc-hc) and [MPC-BE](https://sourceforge.net/projects/mpcbe) and offers a range of settings to tune the quality and presentation of the interpolation.

<div align="center">
  <img width="800" alt="Big Buck Bunny camera pan at 0.25x speed: original 24 fps on the left, HopperRender interpolated to 60 fps on the right" src="https://github.com/user-attachments/assets/543bb7e4-48d0-42c9-b0f6-10fbf37b3929">
  <br>
  <sub>Original 24 fps (left) vs. HopperRender at 60 fps (right), slowed down to 0.25× speed</sub>
</div>

<details>
<summary><strong>Watch the full comparison video (normal speed and 0.25×)</strong></summary>
<br>

https://github.com/user-attachments/assets/87d722d1-2915-4b75-901c-5ab1036622e2

</details>

> [!TIP]
> On Linux? HopperRender is also available as an mpv integration: **[mpv-frame-interpolator](https://github.com/HopperLogger/mpv-frame-interpolator)**.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Building from source](#building-from-source)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Real-time interpolation** of any source frame rate to your monitor's native refresh rate, half of it, an integer multiple of the source rate, or a custom target
- **Bidirectional warping**: frames are warped in both directions and blended for the smoothest result
- **HDR support**, including Dolby Vision metadata passthrough
- **Any resolution**, from DVDs up to 4K Blu-rays
- **Cross-vendor GPU acceleration** via OpenCL on most NVIDIA and AMD graphics cards
- **Adaptive performance**: internal settings adjust automatically to your PC's capabilities
- **Smart activation**: detects the source frame rate (including variable frame rate and playback speed changes) and turns interpolation off when it isn't needed
- **Scene change detection** so cuts don't get interpolated
- **Flow visualization**: HSV and greyscale views show the motion that was detected in a scene
- **System tray icon** for toggling the filter, switching output modes, and viewing live statistics
- **Portable**: no installer and no internet connection required
- **Renderer-agnostic**: works with madVR, MPC Video Renderer, Enhanced Video Renderer, and more

## Requirements

| Component | Requirement |
| --- | --- |
| OS | Windows 10 or later |
| GPU | NVIDIA, AMD or Intel GPU with an OpenCL 1.2+ capable driver |
| Player | A DirectShow media player such as [MPC-HC](https://github.com/clsid2/mpc-hc/releases) or [MPC-BE](https://sourceforge.net/projects/mpcbe) |
| Renderer *(recommended)* | [madVR](https://www.videohelp.com/software/madVR) or [MPC Video Renderer](https://github.com/Aleksoid1978/VideoRenderer). Both are actively maintained and handle HDR properly |

## Installation

1. Download the DLL that matches your media player from the [**Releases**](https://github.com/HopperLogger/HopperRender/releases/latest) page:

   | File | Use it for |
   | --- | --- |
   | `HopperRender.x64.dll` | 64-bit players (e.g. `mpc-hc64.exe`, `mpc-be64.exe`, `PotPlayerMini64.exe`). **This is what most people need.** |
   | `HopperRender.x86.dll` | 32-bit players (e.g. `mpc-hc.exe`, `mpc-be.exe`, `PotPlayerMini.exe`) |

   The DLL must match the player's architecture, not the Windows version. A 32-bit player can't load the 64-bit DLL, and the other way around.
2. Copy the DLL to a permanent location, such as `C:\Program Files\HopperRender\`.
3. Register it as an external filter in your player (MPC-HC / MPC-BE):
   1. Open **Options** → **External Filters**.
   2. Click **Add Filter...**, then **Browse...**, and select the DLL.
   3. Select the newly added HopperRender entry and set it to **Prefer**.
   4. Close the settings and restart the player.

That's it. Videos played in MPC-HC/BE will now be interpolated to your display's refresh rate.

> [!IMPORTANT]
> Don't move or delete the folder containing the DLL after registering it, or the player won't be able to find the filter.
>
> To update HopperRender, close the player and replace the DLL with the new version, using the same file name and location. You don't need to register it again.

<details>
<summary><strong>Show step-by-step screenshots</strong></summary>
<br>

**Open External Filters and click Add Filter:**

<img alt="Add filter" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/ecf7003b-09f1-4195-8c13-7767f195ee62">

**Click Browse:**

<img alt="Browse" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/59b27a26-8889-4549-a67b-bbdec7f7eec6">

**Select HopperRender.x64![Uploading HR-Settings.png…]()
.dll:**

<img alt="Select DLL" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/be6dc452-2f4d-4570-8745-3d49102006ee">

**Set the filter to Prefer:**

<img alt="Prefer filter" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/9cbc1894-8ff3-410a-bd0d-2f9e785bcac3">

</details>

## Configuration

While a video is playing, right-click it in MPC-HC/BE and choose **Filters** → **HopperRender** to open the settings. You can also use the HopperRender **system tray icon** for quick access to common options and statistics.

<div align="center">
  <img alt="HopperRender settings dialog" height="500px" src="https://github.com/user-attachments/assets/6f7d675a-8938-481b-a90e-0507591958aa">
</div>

Settings are saved automatically to `HKEY_CURRENT_USER\Software\HopperRender` and restored the next time the filter is loaded.

### Settings

| Setting | Description |
| --- | --- |
| **On / Off** | Turns interpolation on or off. |
| **Frame Output** | Selects what is displayed (see [Frame output modes](#frame-output-modes)). |
| **Target Frame Rate** | The frame rate to interpolate to: *Display Rate* (default), *1/2x Display Rate*, an integer multiple of the source frame rate (e.g. 48 fps for 24 fps content), or *Custom*. |
| **Delta Scalar** | How strongly frames are moved to follow the on-screen motion. |
| **Neighbor Scalar** | How strongly each region is biased by the motion around it. Higher values give more uniform interpolation but may miss small movements. |
| **Black / White Levels** | Level correction for limited-range sources (e.g. when blacks look grey or whites aren't fully bright). |
| **Scene Change Threshold** | Frame transitions with a difference above this threshold are treated as scene cuts and aren't interpolated. The peak detected value is shown next to the control to help with tuning. |
| **Buffer Frames** | Set to around `15` for live sources (e.g. capture cards) to prevent dropped frames. **Leave at `0` for normal video playback.** |

### Frame output modes

| Mode | Description |
| --- | --- |
| **Warped Frame 1 → 2** | Only the warp from the previous frame to the current frame. |
| **Warped Frame 2 → 1** | Only the warp from the current frame to the previous frame. |
| **Blended Frame** | Both warp directions blended together *(default)*. |
| **Blended Frame (w/o warping)** | Skips optical flow entirely and simply blends the two source frames. |
| **HSV Flow** | Visualizes the optical flow as color, where hue indicates the direction of motion (see the color wheel below). |
| **Grey Flow** | Visualizes the optical flow in greyscale, where brightness indicates the magnitude of motion. |
| **Side-by-side 1** | Original on the left, interpolated on the right (split down the middle). |
| **Side-by-side 2** | Original on the left, interpolated on the right (both scaled down). |

<div align="center">
  <img alt="HSV flow color wheel" height="200px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/b025d4ce-cfa2-4702-b184-2c09f4254246">
</div>

### Status

The status section shows the current state of the filter, the source and target frame rates, the frame and calculation resolutions, and the processing times of the optical flow calculation and the frame warping.

> [!NOTE]
> For smooth playback, the optical flow and warping times **combined** must stay below the source frame time (`1 / source fps`, e.g. 41.7 ms for 24 fps content).

### Advanced: calculation resolution

<details>
<summary>Manually limit the internal optical flow resolution (not recommended)</summary>
<br>

By default, the optical flow is calculated at a resolution of at most 270p. Higher resolutions tend to produce *worse* results, and lower resolutions offer little benefit, so changing this is not recommended. If you want to experiment anyway:

1. Open the Registry Editor and navigate to `HKEY_CURRENT_USER\Software\HopperRender`.
2. Create a new **DWORD (32-bit) Value** named `MaxCalcRes`.
3. Enter the desired maximum height using the **Decimal** base.

The input resolution is repeatedly halved until the frame height is less than or equal to `MaxCalcRes`, so this value is an upper bound rather than an exact resolution. Delete the value to restore the default of 270.

</details>

## How it works

> [!NOTE]
> This is a simplified overview of the interpolation pipeline. Refer to the source code for the full details.

1. **Pre-processing:** The frames are blurred internally (depending on the settings) so that the search ignores pixel-level detail and compression artifacts.
2. **Layered offset search:** An offset array with 5 layers holds different candidate shifts that are evaluated in parallel. Starting with horizontal shifts of `-2, -1, 0, 1, 2`, the first frame is shifted accordingly and compared against the next frame using the luma (Y) channel difference.
3. **Best-match selection:** The absolute pixel deltas are reduced to a single value per layer, and the layer with the lowest difference wins. Depending on which layer wins, the search either moves on to vertical motion or keeps going further in the negative or positive x direction. This repeats until the best offset is found or the step budget runs out.
4. **Hierarchical refinement:** Once the best offset for the whole frame is found, the window size is reduced to a quarter and the search continues for every window from its previous position, down to the individual pixel level if the settings allow.
5. **Bidirectional flow:** The offset array is flipped to get the warp from frame 2 → 1 in addition to frame 1 → 2, and both arrays are blurred for a smoother warp.
6. **Frame synthesis:** Intermediate frames are generated by scaling the offsets to the intermediate time position. Poorly matched pixels get artifact removal, and the two warp directions are blended into the final frame.

## Building from source

### Prerequisites

- [Visual Studio 2026](https://visualstudio.microsoft.com/) with the **Desktop development with C++** workload (the project uses the `v145` platform toolset and the Windows SDK included with the workload)

No GPU SDK is needed. The [Khronos OpenCL headers](https://github.com/KhronosGroup/OpenCL-Headers) are included in [`common/OpenCL`](common/OpenCL), and the OpenCL API is loaded at runtime from the `OpenCL.dll` installed with the GPU driver.

### Build

```sh
git clone https://github.com/HopperLogger/HopperRender.git
```

1. Open `HopperRender.sln` in Visual Studio.
2. Select the **Release** configuration and the platform:
   - **x64** for 64-bit players (most setups)
   - **Win32** for 32-bit players
3. Build the solution (**Build** → **Build Solution**).
4. The compiled DLL is placed in `x64\Release\HopperRender.dll` or `Win32\Release\HopperRender.dll`.

## Troubleshooting

- **No interpolation / filter shows as inactive:** Make sure your GPU driver supports OpenCL 1.2 or newer. If OpenCL fails to initialize, the settings dialog, tray icon, and a Windows notification will tell you, and the video plays without interpolation.
- **Stuttering or dropped frames:** Check the calculation times in the status section. If they come close to the source frame time, your GPU can't keep up at the current settings.
- **Filter doesn't show up or isn't used:** Make sure you downloaded the DLL that matches your player's architecture (x64 for 64-bit players, x86 for 32-bit players).
- **Log files:** Each session writes a log to your temp folder (`%TEMP%\HopperRender_<timestamp>.log`). Please attach it when [reporting a bug](https://github.com/HopperLogger/HopperRender/issues/new/choose).

## Contributing

Bug reports, feature requests, and pull requests are welcome! Please read the [contributing guide](CONTRIBUTING.md) to learn how to report issues, set up a development build, and submit changes. By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

HopperRender is licensed under the [GNU General Public License v3.0](LICENSE).

## Acknowledgements

- Based on Microsoft's [EZRGB24 Filter Sample](https://learn.microsoft.com/en-us/windows/win32/directshow/ezrgb24-filter-sample).
- DirectShow core parts were inspired by [LAV Filters](https://github.com/clsid2/LAVFilters).
- Dolby Vision metadata passthrough was adapted from [MPC Video Renderer](https://github.com/Aleksoid1978/VideoRenderer).
- Demo footage: [Big Buck Bunny](https://peach.blender.org/) © Blender Foundation | www.bigbuckbunny.org, licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).