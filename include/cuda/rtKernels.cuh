#ifndef CUDA_RT_KERNELS_CUH
#define CUDA_RT_KERNELS_CUH

#include <cuda_runtime.h>

#include "RT/vec3.h"
#include "RT/hitable.h"

__global__ void launchRays(vec3* hitPos, vec3* hitNormal, float* hitDist,
                           int* hitFlag, int nx, int ny, vec3 llc, vec3 horiz,
                           vec3 vert, vec3 rayDir, hitable** world);

__global__ void launchRaysMultiBounce(vec3* hitPos, vec3* hitNormal,
                                      float* hitDist, int* hitCount, int nx,
                                      int ny, vec3 llc, vec3 horiz, vec3 vert,
                                      vec3 rayDir, hitable** world,
                                      int maxBounces);

#endif
