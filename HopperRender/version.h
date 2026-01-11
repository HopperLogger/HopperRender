#pragma once

// Version information
#define VERSION_MAJOR 2
#define VERSION_MINOR 0
#define VERSION_PATCH 2
#define VERSION_BUILD 4

#define VERSION_STRING_WITH_NAME "HopperRender V2.0.2.4"

// Macro to stringify for use in code
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// String representation
#define VERSION_STRING TOSTRING(VERSION_MAJOR) "." TOSTRING(VERSION_MINOR) "." TOSTRING(VERSION_PATCH) "." TOSTRING(VERSION_BUILD)
