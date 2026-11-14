// Runtime loader for the OpenCL API.
//
// Instead of linking against an OpenCL.lib import library,
// HopperRender resolves the OpenCL functions it uses at runtime from the system's OpenCL ICD loader (OpenCL.dll).
// This means:
//  - Building only requires the Khronos OpenCL headers in common/OpenCL/include.
//  - The filter can still be loaded when OpenCL.dll is missing; the OpenCL initialization then simply fails
//    with CL_PLATFORM_NOT_FOUND_KHR and the video is played without interpolation.
//  - Missing exports (e.g. on very old OpenCL runtimes) result in an error code instead of a load failure.
//
// When using a new OpenCL function, add it to HR_OPENCL_FUNCTIONS and add a forwarding wrapper below.

#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <CL/cl_function_types.h>
#include <Windows.h>

namespace {

// Returned by clGetPlatformIDs when OpenCL.dll could not be loaded (defined in cl_ext.h)
constexpr cl_int kPlatformNotFound = -1001; // CL_PLATFORM_NOT_FOUND_KHR

// Returned by all other functions when they are not exported by the installed OpenCL.dll
constexpr cl_int kFunctionUnavailable = CL_INVALID_OPERATION;

#define HR_OPENCL_FUNCTIONS(X)    \
    X(clBuildProgram)             \
    X(clCreateBuffer)             \
    X(clCreateCommandQueue)       \
    X(clCreateContext)            \
    X(clCreateKernel)             \
    X(clCreateProgramWithSource)  \
    X(clEnqueueFillBuffer)        \
    X(clEnqueueNDRangeKernel)     \
    X(clEnqueueReadBuffer)        \
    X(clEnqueueWriteBuffer)       \
    X(clFinish)                   \
    X(clGetDeviceIDs)             \
    X(clGetDeviceInfo)            \
    X(clGetEventProfilingInfo)    \
    X(clGetPlatformIDs)           \
    X(clGetProgramBuildInfo)      \
    X(clReleaseCommandQueue)      \
    X(clReleaseContext)           \
    X(clReleaseDevice)            \
    X(clReleaseKernel)            \
    X(clReleaseMemObject)         \
    X(clReleaseProgram)           \
    X(clSetKernelArg)             \
    X(clWaitForEvents)

struct OpenCLApi {
#define HR_DECLARE_POINTER(fn) fn##_t* fn = nullptr;
    HR_OPENCL_FUNCTIONS(HR_DECLARE_POINTER)
#undef HR_DECLARE_POINTER

    OpenCLApi() {
        // Only search System32 (SysWOW64 for 32-bit processes) so a planted OpenCL.dll next to the
        // player or the filter can never be picked up. The module is intentionally never freed.
        HMODULE module = LoadLibraryExW(L"OpenCL.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module) return;

#define HR_RESOLVE_POINTER(fn) fn = reinterpret_cast<fn##_t*>(GetProcAddress(module, #fn));
        HR_OPENCL_FUNCTIONS(HR_RESOLVE_POINTER)
#undef HR_RESOLVE_POINTER
    }
};

// Loaded lazily on the first OpenCL call (never from DllMain); thread-safe since C++11
const OpenCLApi& Api() {
    static const OpenCLApi api;
    return api;
}

// Helper for functions that return an object and report errors through errcode_ret
template <typename T>
T Unavailable(cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = kFunctionUnavailable;
    return nullptr;
}

} // namespace

// Forwarding wrappers. The signatures must match the declarations in CL/cl.h exactly.

CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
    if (auto fn = Api().clGetPlatformIDs) return fn(num_entries, platforms, num_platforms);
    if (num_platforms) *num_platforms = 0;
    return kPlatformNotFound;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) {
    if (auto fn = Api().clGetDeviceIDs) return fn(platform, device_type, num_entries, devices, num_devices);
    if (num_devices) *num_devices = 0;
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (auto fn = Api().clGetDeviceInfo) return fn(device, param_name, param_value_size, param_value, param_value_size_ret);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseDevice(cl_device_id device) {
    if (auto fn = Api().clReleaseDevice) return fn(device);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices,
                void(CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
                void* user_data, cl_int* errcode_ret) {
    if (auto fn = Api().clCreateContext) return fn(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
    return Unavailable<cl_context>(errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context context) {
    if (auto fn = Api().clReleaseContext) return fn(context);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret) {
    if (auto fn = Api().clCreateCommandQueue) return fn(context, device, properties, errcode_ret);
    return Unavailable<cl_command_queue>(errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue command_queue) {
    if (auto fn = Api().clReleaseCommandQueue) return fn(command_queue);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret) {
    if (auto fn = Api().clCreateBuffer) return fn(context, flags, size, host_ptr, errcode_ret);
    return Unavailable<cl_mem>(errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem memobj) {
    if (auto fn = Api().clReleaseMemObject) return fn(memobj);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context context, cl_uint count, const char** strings, const size_t* lengths, cl_int* errcode_ret) {
    if (auto fn = Api().clCreateProgramWithSource) return fn(context, count, strings, lengths, errcode_ret);
    return Unavailable<cl_program>(errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options,
               void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
    if (auto fn = Api().clBuildProgram) return fn(program, num_devices, device_list, options, pfn_notify, user_data);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size,
                      void* param_value, size_t* param_value_size_ret) {
    if (auto fn = Api().clGetProgramBuildInfo) return fn(program, device, param_name, param_value_size, param_value, param_value_size_ret);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program program) {
    if (auto fn = Api().clReleaseProgram) return fn(program);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
    if (auto fn = Api().clCreateKernel) return fn(program, kernel_name, errcode_ret);
    return Unavailable<cl_kernel>(errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) {
    if (auto fn = Api().clSetKernelArg) return fn(kernel, arg_index, arg_size, arg_value);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel kernel) {
    if (auto fn = Api().clReleaseKernel) return fn(kernel);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset,
                       const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list, cl_event* event) {
    if (auto fn = Api().clEnqueueNDRangeKernel)
        return fn(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset,
                    size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (auto fn = Api().clEnqueueFillBuffer)
        return fn(command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void* ptr,
                    cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (auto fn = Api().clEnqueueReadBuffer)
        return fn(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void* ptr,
                     cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (auto fn = Api().clEnqueueWriteBuffer)
        return fn(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue command_queue) {
    if (auto fn = Api().clFinish) return fn(command_queue);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint num_events, const cl_event* event_list) {
    if (auto fn = Api().clWaitForEvents) return fn(num_events, event_list);
    return kFunctionUnavailable;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (auto fn = Api().clGetEventProfilingInfo) return fn(event, param_name, param_value_size, param_value, param_value_size_ret);
    return kFunctionUnavailable;
}
