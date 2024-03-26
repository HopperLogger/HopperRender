# HopperRender
An optical flow frame interpolator with DirectShow integration that allows you to watch any local media file in 60 fps.
This is my first DirectShow and CUDA project. The goal is to achieve pretty decent frame interpolation with a variety of user customizable settings.
The filter can be added to a DirectShow media player like MPC-HC or MPC-BE in combination with madVR as the renderer.
Please keep in mind that this project is still in ongoing development and there are very likely some bugs depending on the environment you're running and the setting you use.

## Features
[To be added]

## How to get started?

### Installation
1. Open MPC-HC or MPC-BE's settings.
2. Select the Externel Filters submenu.
4. Click on Add Filter.
![Setup1](https://github.com/HopperLogger/HopperRender/assets/121826818/ecf7003b-09f1-4195-8c13-7767f195ee62)
6. Click on Browse.
![Setup2](https://github.com/HopperLogger/HopperRender/assets/121826818/59b27a26-8889-4549-a67b-bbdec7f7eec6)
8. Select the HopperRender.dll file.
![Setup3](https://github.com/HopperLogger/HopperRender/assets/121826818/e059abd1-1dab-4674-b0bb-4311ac760b54)
10. Select Prefer.
11. Close the settings and the media player.
![Setup4](https://github.com/HopperLogger/HopperRender/assets/121826818/9cbc1894-8ff3-410a-bd0d-2f9e785bcac3)

That's it! You can now play a video with MPC-HC/BE and HopperRender will interpolate it to 60fps.

### Settings
You can access the settings when playing back a video with HopperRender by right clicking on the video in MPC-HC/BE, selecting the Filters menu and then HopperRender.

<img width="320" alt="Properties" src="https://github.com/HopperLogger/HopperRender/assets/121826818/03309480-1bb8-40d5-a38a-641af109b3e0">

 - You can activate and deactivate the interpolation
 - You can select which type of frame output you want to see
 - *Warped Frame 1 -> 2: Shows just the warping from the previous to the current frame*
 - *Warped Frame 2 -> 1: Shows just the warping from the current to the previous frame*
 - *Blended Frame: Blends both warp directions together*
 - *HSV Flow: Visualizes the optical flow as a color representation, where the color indicates the direction of movement*
<img src="https://github.com/HopperLogger/HopperRender/assets/121826818/b025d4ce-cfa2-4702-b184-2c09f4254246" alt="color-circle" width="300">
 - *Blurred Frames: Outputs the blurred source frames*

 - You can set the number of iterations (0 will automatically do as many iterations as possible)
 - You can set the Frame and Flow blur kernel sizes which controls how much the frames or the flow will be blurred
 - In the status section, you can see the current state of HopperRender, the number of calculation steps that are currently performed, the source framerate, as well as the frame and calculation resolutions
 - The settings will be automatically saved to the registry (HKEY_CURRENT_USER\Software\HopperRender) so next time the filter is used, it loads the settings automatically
