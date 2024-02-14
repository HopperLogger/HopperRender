#ifndef __IEZ__
#define __IEZ__

#ifdef __cplusplus
extern "C" {
#endif

    // { fd5010a3-8ebe-11ce-8183-00aa00577da1 }
    DEFINE_GUID(IID_SettingsInterface,
        0xfd5010a3, 0x8ebe, 0x11ce, 0x81, 0x83, 0x00, 0xaa, 0x00, 0x57, 0x7d, 0xa1);

    DECLARE_INTERFACE_(SettingsInterface, IUnknown)
    {
        STDMETHOD(get_Settings) (THIS_
            bool* pbActivated,       // Is the filter activated
            int* piNumIterations,    // Number of iterations to find the ideal offset
            int* piMaxOffsetDivider, // The divider used to calculate the initial global offset
            int* piIntActiveState,   // Is the effect active
            double* pdSourceFPS,     // The source frames per second
            int* piNumSteps          // Number of steps executed to find the ideal offset (limits the maximum offset)
            ) PURE;

        STDMETHOD(put_Settings) (THIS_
            bool bActivated,        // Is the filter activated
            int iNumSteps,          // Number of steps executed to find the ideal offset (limits the maximum offset)
            int iMaxOffsetDivider   // The divider used to calculate the initial global offset
            ) PURE;
    };

#ifdef __cplusplus
}
#endif

#endif // __IEZ__