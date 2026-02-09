#ifndef CUDA_RT_KERNELS_CUH
#define CUDA_RT_KERNELS_CUH

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#include "hip/hipifly.hpp"
#else
#include <cuda_runtime.h>
#endif

#include "RT/vec3.h"
#include "scene/bvh.h"

__global__ void launchRaysMultiBounce(vec3* __restrict__ hitNormal,
                                      vec3* __restrict__ lastDir,
                                      Real* __restrict__ hitDist, int* __restrict__ hitCount, int nx,
                                      int ny, vec3 llc, vec3 horiz, vec3 vert,
                                      vec3 rayDir, const Triangle* __restrict__ triangles,
                                      const BvhNode* __restrict__ nodes, int rootIndex,
                                      int maxBounces);

#endif
