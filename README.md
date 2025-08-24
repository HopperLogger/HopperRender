<div align="center">
  <img alt="logo" height="200px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/faa253f0-3276-4404-aa4a-bb9c2b35056c">
</div>

# HopperRender
An optical flow frame interpolator with DirectShow integration that allows you to watch any local media file at the native refresh rate of your monitor.
This is my first project using DirectShow and CUDA, which has now transitioned to OpenCL for better cross-platform compatibility. The goal is to achieve pretty decent frame interpolation with a variety of user customizable settings.
The filter can be added to a DirectShow media player like MPC-HC or MPC-BE.
> Please note that this project is no longer in development and is superseded by my [mpv frame interpolator](https://github.com/HopperLogger/mpv-frame-interpolator). However, I will occasionally port significant improvements over to this project.

## Features
- Realtime frame interpolation of any source framerate to the native refresh rate of your monitor
- Compatible with HDR video _(usage of madVR recommended)_
- All resolutions _(even DVDs and 4K Blu-rays)_ are supported
- No installation or internet connection required
- Cross-platform compatible with most NVIDIA and AMD cards
- Warps frames in both directions and blends them for the smoothest experience
- User customizable settings to adjust the quality and presentation of the interpolation
- HSV Flow visualization lets you see the calculated movements of objects in a scene
- Automatically adjusts internal settings to match the PC's performance
- Compatible with madVR, Enhanced Video Renderer, MPC-Video Renderer, and more
- Automatically detects the source frame rate (as well as playback speed) and disables interpolation if not needed
- Small Exporter GUI lets you render videos files with HopperRender

## How to get started?
To use this filter, you need to use a DirectShow player like [MPC-HC](https://github.com/clsid2/mpc-hc/releases) or [MPC-BE](https://sourceforge.net/projects/mpcbe).

If you decide to use MPC-BE, please install [LAVFilters](https://github.com/Nevcairiel/LAVFilters/releases) and add the _LAV Video Filter_ to your filters in the player settings. This will ensure proper compatibility with HopperRender.

The usage of [madVR](https://www.videohelp.com/software/madVR) is recommended, but not necessary.

### Installation
1. Open MPC-HC or MPC-BE's settings.
2. Select the External Filters submenu.
3. Click on Add Filter.
<div align="left">
  <img alt="install1" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/ecf7003b-09f1-4195-8c13-7767f195ee62">
</div>
4. Click on Browse.
<div align="left">
  <img alt="install2" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/59b27a26-8889-4549-a67b-bbdec7f7eec6">
</div>
5. Select the HopperRender.dll file from the extracted zip file.
<div align="left">
  <img alt="install3" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/be6dc452-2f4d-4570-8745-3d49102006ee">
</div>
<div>6. Select Prefer.</div>
7. Close the settings and the media player.
<div align="left">
  <img alt="install4" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/9cbc1894-8ff3-410a-bd0d-2f9e785bcac3">
</div>

That's it! You can now play a video with MPC-HC/BE and HopperRender will interpolate it to 60fps or more.

> Note: Do not move or delete the folder containing the HopperRender.dll file, otherwise the media player won't find it.

### Settings
You can access the settings when playing back a video with HopperRender by right clicking on the video in MPC-HC/BE, selecting the Filters menu and then HopperRender.

<div align="center">
  <img alt="properties" height="300px" src="https://github.com/user-attachments/assets/be4efd96-9c23-4094-8beb-853ac2a2d707">
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

    - _Grey Flow: Visualizes the optical flow as a black and white representation, where the brightness indicates the magnitude of movement_
    - _Side-by-side 1: Shows the difference between no interpolation on the left, and interpolation on the right (split in the middle)_
    - _Side-by-side 2: Shows the difference between no interpolation on the left, and interpolation on the right (scaled down side by side)_
- You can set a custom target fps that should be interpolated to. Setting this to 0 will automatically use the display refresh rate.
- You can set the Delta Scalar (controls how much the filter will try to move the frames to simulate the motion on screen)
- You can set the Neighbor Scalar (controls how much the filter will be biased by the surrounding motion, higher values will lead to more uniform interpolation but might miss small motions)
- You can set the Black and White levels which allows for level correction if the input has limited range (i.e. the blacks look grey, and the whites are not full brightness)
- In the status section, you can see the current state of HopperRender, the source and target framerate, the frame and calculation resolutions, as well as the calculation times of the optical flow calulcation and frame warping pipeline *(both times combined should always be lower than 1/source framerate!)*
- The settings will be automatically saved to the registry `HKEY_CURRENT_USER\Software\HopperRender` so next time the filter is used, it loads the settings automatically

## How it works
> Note: The following is a very brief overview of how the filter works. It is not a complete, or 100% accurate description of the interpolation process. Refer to the source code for more details.

- To prevent the algorithm from focusing on pixel level details or compression artifacts, we first (depending on the user setting) blur the frames internally to use for the optical flow calculation
- HopperRender uses an offset array that shifts the frame according to the values contained in it
- The offset array has 5 layers that contain different shifts that can be 'tried out' at the same time to find the best one
- The first step involves setting the 5 layers to a horizontal shift of -2, -1, 0, 1, and 2
- Then, the first frame is shifted accordingly and we subtract the y-channel difference of the shifted frame to the unshifted next frame
- We then reduce all the absolute pixel deltas to one value and find out which layer (i.e. which horizontal shift) contains the lowest value and therefore difference
- Depending on the resulting layer index, we can either move on to the same procedure for the vertical movement, or continue moving in the negative or positive x direction
- We repeat this process until we are certain we found the best offset, or are out of calculation steps
- After having found the best offset for the entire frame, we decrease our window size to a quarter the size and continue the search again for every individual window starting at the previous position
- Depending on the user setting, we do this until we get to the individual pixel level
- Finally, we flip the offset array to give us not just the ideal shift to warp frame 1 to frame 2, but also to warp frame 2 to frame 1
- We then blur both offset arrays depending on the user settings to get a more smooth warp
- Then we use these offset arrays to generate intermediate frames by multiplying the offset values by certain scalars
- We add a bit of artifact removal for the pixels that weren't ideally moved and blend the warped frames from both directions together

## Exporter
The exporter allows you to interpolate video files with custom settings instead of just watching it with the DirectShow Filter.
> Note: Showing the preview will negatively impact export performance and will not show a frame time accurate playback. The Exporter does not support HDR videos and will currently not encode any audio.
<div align="center">
    <img alt="exporter" height="400px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/2c917bec-4b8a-43a4-a9e6-7835743b5b01">
    </div>

### Installation
1. Download [OpenCV](https://github.com/opencv/opencv/releases) and extract it to `C:\opencv`.
2. Add `C:\opencv\build\x64\vc16\bin` to the Path System/Enviornment Variable.
3. Download [OpenH264 1.8.0](https://github.com/cisco/openh264/releases/download/v1.8.0/openh264-1.8.0-win64.dll.bz2), extract it and copy the .dll to `C:\opencv\build\x64\vc16\bin`.
4. In the extracted release, launch `HopperRenderExporter.exe` and select the video you want to interpolate, as well as the desired settings.

## Acknowledgements

This project is based on the [EZRGB24 Filter Sample](https://learn.microsoft.com/en-us/windows/win32/directshow/ezrgb24-filter-sample) and the DirectShow core parts were inspired by the [LAV Video Decoder](https://github.com/clsid2/LAVFilters).
