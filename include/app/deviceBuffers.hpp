#ifndef APP_DEVICE_BUFFERS_HPP
#define APP_DEVICE_BUFFERS_HPP

#include "RT/vec3.h"        // includes precision.h → Real, cuRealComplex

struct deviceBuffers {
    vec3* hitPos = nullptr;
    vec3* hitNormal = nullptr;
    vec3* lastDir = nullptr;        // Last-bounce incoming ray direction
    Real* hitDist = nullptr;
    int* hitFlag = nullptr;
    int* hitCount = nullptr;        // Device memory
    int* hitCountHost = nullptr;    // Host-pinned memory for stats
    cuRealComplex* accum = nullptr;      // Device memory for atomic ops
    cuRealComplex* accumHost = nullptr;  // Host-pinned memory for reading results
    int count = 0;
};

void allocateDeviceBuffers(deviceBuffers& buffers, int count);
void freeDeviceBuffers(deviceBuffers& buffers);

#endif
