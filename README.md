<div align="center">
  <img alt="logo" height="200px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/faa253f0-3276-4404-aa4a-bb9c2b35056c">
</div>

# HopperRender
An optical flow frame interpolator with DirectShow integration that allows you to watch any local media file in 60 fps.
This is my first DirectShow and CUDA project. The goal is to achieve pretty decent frame interpolation with a variety of user customizable settings.
The filter can be added to a DirectShow media player like MPC-HC or MPC-BE.
> Please keep in mind that this project is still in ongoing development and there are very likely some bugs depending on the environment you're running and the setting you use. The interpolation quality is also not perfect yet, but pretty decent most of the time, especially for 24 fps -> 60 fps conversion.

## Features
- Realtime frame interpolation of any source framerate to 60fps
- Compatible with HDR video _(usage of madVR recommended)_
- All resolutions _(even DVDs and 4K Blu-rays)_ are supported
- No installation or internet connection required
- Warps frames in both directions and blends them for the smoothest experience
- User customizable settings to adjust the quality and performance of the interpolation
- HSV Flow visualization lets you see the calculated movements of objects in a scene
- Automatically adjusts internal settings to match the PC's performance
- Compatible with madVR, Enhanced Video Renderer, MPC-Video Renderer, and more
- Automatically detects the source frame rate (as well as playback speed) and disables interpolation if not needed

## How to get started?
To use this filter, you need to use a DirectShow player like [MPC-HC](https://github.com/clsid2/mpc-hc/releases) or [MPC-BE](https://sourceforge.net/projects/mpcbe).
This filter uses the CUDA API, so it requires a NVIDIA GPU (GTX 950 or newer).
The usage of [madVR](https://www.videohelp.com/software/madVR) is recommended, but not necessary.

### Installation
1. Open MPC-HC or MPC-BE's settings.
2. Select the External Filters submenu.
4. Click on Add Filter.
<div align="left">
  <img alt="install1" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/ecf7003b-09f1-4195-8c13-7767f195ee62">
</div>
5. Click on Browse.
<div align="left">
  <img alt="install2" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/59b27a26-8889-4549-a67b-bbdec7f7eec6">
</div>
6. Select the HopperRender.dll file from the extracted zip file.
<div align="left">
  <img alt="install3" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/e059abd1-1dab-4674-b0bb-4311ac760b54">
</div>
7. Select Prefer.
8. Close the settings and the media player.
<div align="left">
  <img alt="install4" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/9cbc1894-8ff3-410a-bd0d-2f9e785bcac3">
</div>

That's it! You can now play a video with MPC-HC/BE and HopperRender will interpolate it to 60fps.

> Note: Do not move or delete the folder containing the HopperRender.dll file, otherwise the media player won't find it.

### Settings
You can access the settings when playing back a video with HopperRender by right clicking on the video in MPC-HC/BE, selecting the Filters menu and then HopperRender.

<div align="center">
  <img alt="properties" height="300px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/03309480-1bb8-40d5-a38a-641af109b3e0">
</div>

- You can activate and deactivate the interpolation
- You can select which type of frame output you want to see:
    - _Warped Frame 1 -> 2: Shows just the warping from the previous to the current frame_
    - _Warped Frame 2 -> 1: Shows just the warping from the current to the previous frame_
    - _Blended Frame: Blends both warp directions together_
    - _HSV Flow: Visualizes the optical flow as a color representation, where the color indicates the direction of movement_

    <div align="center">
    <img alt="color-circle" height="200px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/b025d4ce-cfa2-4702-b184-2c09f4254246">
    </div>

    - _Blurred Frames: Outputs the blurred source frames_
- You can set the number of iterations (0 will automatically do as many iterations as possible)
- You can set the Frame and Flow blur kernel sizes which controls how much the frames or the flow will be blurred
- In the status section, you can see the current state of HopperRender, the number of calculation steps that are currently performed, the source framerate, as well as the frame and calculation resolutions
- The settings will be automatically saved to the registry `HKEY_CURRENT_USER\Software\HopperRender` so next time the filter is used, it loads the settings automatically

## Acknowledgements

This project is based on the [EZRGB24 Filter Sample](https://learn.microsoft.com/en-us/windows/win32/directshow/ezrgb24-filter-sample) and the DirectShow core parts were inspired by the [LAV Video Decoder](https://github.com/clsid2/LAVFilters).
