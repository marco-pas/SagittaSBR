#ifndef SCENE_WORLD_CUH
#define SCENE_WORLD_CUH

#include "RT/hitable.h"
#include "RT/vec3.h"

__global__ void createWorld(hitable** deviceList, hitable** deviceWorld);
__global__ void freeWorld(hitable** deviceList, hitable** deviceWorld);
__global__ void createWorldFromMesh(hitable** deviceList, hitable** deviceWorld,
                                    const vec3* vertices, const int* indices,
                                    int triangleCount);
__global__ void freeWorldFromMesh(hitable** deviceList, hitable** deviceWorld,
                                  int triangleCount);

#endif
