#ifndef CUDA_PO_KERNELS_CUH
#define CUDA_PO_KERNELS_CUH

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#else
#include <cuComplex.h>
#endif

#include "RT/vec3.h"

__global__ void integratePo(vec3* hitPos, vec3* hitNormal, float* hitDist,
                            int* hitFlag, int n, float k, vec3 kInc,
                            float rayArea, cuFloatComplex* accum);

__global__ void integratePoMultiBounce(vec3* hitPos, vec3* hitNormal,
                                       float* hitDist, int* hitCount, int n,
                                       float k, vec3 kInc, float rayArea,
                                       cuFloatComplex* accum,
                                       float reflectionConst);

#endif
