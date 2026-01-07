/*
 * Implementation of scene geometry setup kernels.
 */

#include "world_setup.cuh"
#include "sphere.h"

__global__ void create_world(hitable** d_list, hitable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Center our object at (0.0f, 0.0f, 0.0f)
        d_list[0] = new sphere(vec3(0.0f, 0.0f, 0.0f), 1.0f); // center coordinates + radius of the sphere

        *d_world  = new hitable_list(d_list, 1); // change number of hitables here

        // Example: Multiple spheres configuration (commented out)
        // d_list[0] = new sphere(vec3(1.2f, 0.0f, 0.0f), 0.5f);
        // d_list[1] = new sphere(vec3(-1.2f, 0.0f, 0.0f), 0.5f);
        // *d_world  = new hitable_list(d_list, 2);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world) {
    delete d_list[0]; 
    delete *d_world;
}
