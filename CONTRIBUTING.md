# Contributing to HopperRender

Thanks for your interest in improving HopperRender! Bug reports, feature ideas, and pull requests are all welcome.

By participating in this project, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and requesting features

Please use the [issue forms](https://github.com/HopperLogger/HopperRender/issues/new/choose):

- **Bug report:** for crashes, freezes, visual artifacts, or anything else that doesn't work as expected. Please include the log file (`%TEMP%\HopperRender_<timestamp>.log`), because most problems can't be diagnosed without it.
- **Feature request:** for new features or improvements.
- **Question:** if you're unsure how something works. Check the [README](README.md) first.

Before opening a new issue, search the [existing issues](https://github.com/HopperLogger/HopperRender/issues?q=is%3Aissue) to see if it has already been reported.

## Pull requests

For small fixes (typos, obvious bugs), feel free to open a pull request directly. For larger changes or new features, please **open an issue first** so we can agree on the approach before you spend time on it.

### Development setup

See [Building from source](README.md#building-from-source) in the README. You only need Visual Studio 2026 with the **Desktop development with C++** workload. No GPU SDK is required.

| Configuration | Use it for |
| --- | --- |
| **Release** | Builds you test or share. This is what releases are built with. |
| **DirectShow** | Debugging inside a media player (unoptimized, with debug info). |

To debug in a player, open the HopperRender project properties, go to **Debugging**, and set **Command** to the player executable (e.g. `C:\Program Files\MPC-HC\mpc-hc64.exe`). Make sure the player uses the HopperRender DLL from your build output folder. This setting is stored in `HopperRender.vcxproj.user`, which is not tracked by git.

### Project layout

| Path | Contents |
| --- | --- |
| `HopperRender/HopperRender.cpp` | The DirectShow filter: media type negotiation, frame timing, and output |
| `HopperRender/HopperRenderSettings.cpp` | The settings dialog |
| `HopperRender/TrayIcon.cpp` | The system tray icon |
| `HopperRender/opticalFlowCalc*.cpp` | The OpenCL optical flow pipeline (shared code, plus SDR and HDR variants) |
| `HopperRender/*Kernel{SDR,HDR}.h` | The OpenCL kernels. If you change one, update the SDR and HDR variants together. |
| `HopperRender/OpenCLLoader.cpp` | Loads the OpenCL API at runtime. If you use a new OpenCL function, add it here. |
| `common/baseclasses` | Microsoft's DirectShow base classes |
| `common/OpenCL` | The Khronos OpenCL headers |

### Code style

- Format your code with the [`.clang-format`](.clang-format) file in the repository root. Visual Studio picks it up automatically. Format only the lines you change, so diffs stay easy to review.
- The project builds without compiler warnings. Please keep it that way for both **x64** and **Win32**.
- Keep changes focused. Unrelated refactoring or reformatting belongs in a separate pull request.

### Testing

There are no automated tests, so please test your change by playing videos with HopperRender:

- In at least one supported player (MPC-HC, MPC-BE, or PotPlayer)
- With SDR content and, if your change touches the frame pipeline, with HDR content
- In both the **x64** and **Win32** builds if your change could behave differently between them

Mention in the pull request what you tested (player, renderer, GPU, and the kind of content).

### Commit messages

Write short summaries in the past tense, describing the change from a user's point of view. Join multiple changes with ` + `:

```
Added a "1/2x Display Rate" target frame rate option
Fixed playback freezing when resuming after a long pause + Improved OpenCL error reporting
```

### Version and changelog

Don't change the version number in `HopperRender/version.h` or edit `CHANGELOG.md` in your pull request. Describe the user-visible changes in the pull request description instead. The maintainer will add them to the changelog when preparing the next release.

## License

HopperRender is licensed under the [GNU General Public License v3.0](LICENSE). By submitting a contribution, you agree that it will be licensed under the same terms.
