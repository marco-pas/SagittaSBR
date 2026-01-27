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

__global__ void launchRaysMultiBounce(vec3* hitPos, vec3* hitNormal,
                                      float* hitDist, int* hitCount, int nx,
                                      int ny, vec3 llc, vec3 horiz, vec3 vert,
                                      vec3 rayDir, const Triangle* triangles,
                                      const BvhNode* nodes, int rootIndex,
                                      int maxBounces);

#endif
