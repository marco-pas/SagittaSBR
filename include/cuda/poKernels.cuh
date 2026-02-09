#ifndef CUDA_PO_KERNELS_CUH
#define CUDA_PO_KERNELS_CUH

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include "hip/hipifly.hpp"
#else
#include <cuComplex.h>
#endif

#include "RT/vec3.h"

__global__ void integratePo(vec3* hitPos, vec3* hitNormal, Real* hitDist,
                            int* hitFlag, int n, Real k, vec3 kInc,
                            Real rayArea, cuRealComplex* accum);

__global__ void integratePoMultiBounce(
    const vec3* __restrict__ hitPos,
    const vec3* __restrict__ hitNormal,
    const vec3* __restrict__ lastDir,
    const Real* __restrict__ hitDist,
    const int* __restrict__ hitCount,
    int n, Real k, vec3 kInc, Real rayArea,
    cuRealComplex* __restrict__ accum,
    Real reflectionConst);

#endif
