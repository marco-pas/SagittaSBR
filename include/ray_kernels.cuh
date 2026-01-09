/*
 * CUDA kernels for ray tracing and intersection testing.
 * Includes single-bounce and multi-bounce ray launching kernels.
 */

#ifndef RAY_KERNELS_CUH
#define RAY_KERNELS_CUH

#include "vec3.h"
#include "hitable_list.h"

// Single-bounce ray launcher kernel
// Shoots rays from a grid and records first hit for each ray
__global__ void launcher_singlebounce(
    vec3* hit_pos,      // Output: hit positions
    vec3* hit_normal,   // Output: hit normals
    float* hit_dist,    // Output: hit distances
    int* hit_flag,      // Output: hit flags (1 if hit, 0 if miss)
    int nx,             // Grid width
    int ny,             // Grid height
    vec3 llc,           // Lower left corner of ray grid
    vec3 horiz,         // Horizontal vector spanning grid
    vec3 vert,          // Vertical vector spanning grid
    vec3 ray_dir,       // Ray direction
    hitable** world     // Scene geometry
);

// Multi-bounce ray launcher kernel
// Shoots rays and traces multiple reflections up to max_bounces
__global__ void launcher(
    vec3* hit_pos,      // Output: first hit positions
    vec3* hit_normal,   // Output: first hit normals
    float* hit_dist,    // Output: cumulative hit distances
    int* hit_count,     // Output: number of bounces per ray
    int nx,             // Grid width
    int ny,             // Grid height
    vec3 llc,           // Lower left corner of ray grid
    vec3 horiz,         // Horizontal vector spanning grid
    vec3 vert,          // Vertical vector spanning grid
    vec3 ray_dir,       // Ray direction
    hitable** world,    // Scene geometry
    int max_bounces     // Maximum number of bounces to trace
);


// Multi-bounce ray launcher kernel (Vector/Polarization version)
// Traces rays and updates the E-field polarization vector at every reflection.
__global__ void launcher_vector(
    vec3* hit_pos,      // Output: 3D position of the ray's final exit point
    vec3* hit_normal,   // Output: Surface normal at the final exit point
    vec3* hit_dir,      // Output: Final propagation direction of the ray (after last bounce)
    vec3* hit_pol,      // Output: Final E-field polarization vector (after all rotations/reflections)
    float* hit_dist,    // Output: Total cumulative path length (Source -> Hit1 -> ... -> HitN)
    int* hit_count,     // Output: Total number of bounces (used for reflection loss/phase flip)
    int nx,             // Input: Grid width
    int ny,             // Input: Grid height
    vec3 llc,           // Input: Origin of the ray grid
    vec3 u,             // Input: Grid horizontal span vector
    vec3 v,             // Input: Grid vertical span vector (often used as initial E-field orientation)
    vec3 ray_dir,       // Input: Initial propagation direction
    hitable **world,    // Input: Scene geometry
    int max_bounces     // Input: Maximum allowed reflections
);

#endif // RAY_KERNELS_CUH
