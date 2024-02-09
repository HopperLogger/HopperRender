#ifndef __IEZ__
#define __IEZ__

#ifdef __cplusplus
extern "C" {
#endif

    // { fd5010a3-8ebe-11ce-8183-00aa00577da1 }
    DEFINE_GUID(IID_IIPEffect,
        0xfd5010a3, 0x8ebe, 0x11ce, 0x81, 0x83, 0x00, 0xaa, 0x00, 0x57, 0x7d, 0xa1);

    DECLARE_INTERFACE_(IIPEffect, IUnknown)
    {
        STDMETHOD(get_IPEffect) (THIS_
            int* effectNum,         // The current effect
            int* numSteps,          // Number of steps executed to find the ideal offset (limits the maximum offset)
            int* maxOffsetDivider   // The divider used to calculate the initial global offset
            ) PURE;

        STDMETHOD(put_IPEffect) (THIS_
            int effectNum,          // Change to this effect
            int numSteps,           // Number of steps executed to find the ideal offset (limits the maximum offset)
            int maxOffsetDivider    // The divider used to calculate the initial global offset
            ) PURE;
    };

#ifdef __cplusplus
}
#endif

#endif // __IEZ__