/*
 * Scene geometry setup and cleanup kernels.
 * Creates and destroys hitable objects on the GPU.
 */

#ifndef WORLD_SETUP_CUH
#define WORLD_SETUP_CUH

#include "hitable_list.h"

// Create scene geometry on GPU
// Currently creates a sphere at origin; can be extended for more complex scenes
__global__ void create_world(hitable** d_list, hitable** d_world);

// Free scene geometry on GPU
__global__ void free_world(hitable** d_list, hitable** d_world);

#endif // WORLD_SETUP_CUH
