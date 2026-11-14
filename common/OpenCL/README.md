# OpenCL Headers

The headers in `include/CL/` are an unmodified subset of the official
[Khronos OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers),
tag **v2026.05.29**, licensed under the Apache License 2.0 (see `LICENSE`).

Only the headers required to build HopperRender are included:

- `cl.h`
- `cl_platform.h`
- `cl_version.h`
- `cl_function_types.h`

HopperRender does not link against `OpenCL.lib`. The OpenCL API is resolved at runtime from the
system's `OpenCL.dll` by `HopperRender/OpenCLLoader.cpp`.

To update, copy the files above from a newer OpenCL-Headers release and update the tag in this file.
