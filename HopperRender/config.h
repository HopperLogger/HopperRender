#pragma once

// Quality Adjustments
#define MAX_CALC_RES 270 // The maximum resolution used to calculate the optical flow

#define NUM_ITERATIONS 0 // How many times the window size used in the ofc calculation will be halved. This controls how precise the optical flow gets. (0: As often as possible)

#define MIN_SEARCH_RADIUS 5  // The minimum window size used in the ofc calculation
#define MAX_SEARCH_RADIUS 16 // The maximum window size used in the ofc calculation

// Performance Adjustments
#define AUTO_SEARCH_RADIUS_ADJUST 1 // Whether to automatically reduce/increase the number of calculation steps and window size depending on performance (0: Disabled, 1: Enabled)

#define UPPER_PERF_BUFFER 1.4 // The upper performance buffer, i.e. calc_time * upper_buffer > frame_time triggers quality reduction
#define LOWER_PERF_BUFFER 1.6 // The lower performance buffer, i.e. calc_time * lower_buffer < frame_time triggers quality improvement

#define CALC_TIME_INTERVAL 240 // The time interval (in frames) in which the average and peak ofc calculation time is calculated

// Debugging
#define INC_APP_IND 1 // Whether or not to include the AppIndicator (0: Disabled, 1: Enabled)
#define SAVE_STATS 0  // Whether or not to save the ofc Calc Times to a log file (0: Disabled, 1: Enabled)

// Default Settings
#define DEFAULT_DELTA_SCALAR 8    // The default delta scalar used in the ofc calculation
#define DEFAULT_NEIGHBOR_SCALAR 6 // The default neighbor scalar used in the ofc calculation
#define DEFAULT_BLACK_LEVEL 0 // The default black level for the output
#define DEFAULT_WHITE_LEVEL 255 // The default white level for the output
#define DEFAULT_SCENE_CHANGE_THRESHOLD 500 // The default scene change threshold
#define DEFAULT_BUFFER_FRAMES 0 // The default number of additional frames to buffer at the start