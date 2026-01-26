#ifndef APP_DEVICE_BUFFERS_HPP
#define APP_DEVICE_BUFFERS_HPP

#if defined(USE_HIP)
#include <hip/hip_complex.h>
#define cuFloatComplex hipFloatComplex
#else
#include <cuComplex.h>
#endif

#include "RT/vec3.h"

struct deviceBuffers {
    vec3* hitPos = nullptr;
    vec3* hitNormal = nullptr;
    float* hitDist = nullptr;
    int* hitFlag = nullptr;
    int* hitCount = nullptr;
    cuFloatComplex* accum = nullptr;
    int count = 0;
};

void allocateDeviceBuffers(deviceBuffers& buffers, int count);
void freeDeviceBuffers(deviceBuffers& buffers);

#endif
