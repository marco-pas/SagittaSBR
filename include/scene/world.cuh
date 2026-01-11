#ifndef SCENE_WORLD_CUH
#define SCENE_WORLD_CUH

#include "RT/hitable.h"

__global__ void createWorld(hitable** deviceList, hitable** deviceWorld);
__global__ void freeWorld(hitable** deviceList, hitable** deviceWorld);

#endif
