/*
 * Physical Optics (PO) integration kernels for RCS computation.
 * Computes scattered field contributions from ray hits using PO approximation.
 */

#ifndef PO_KERNELS_CUH
#define PO_KERNELS_CUH

#include <cuComplex.h>
#include "vec3.h"

// Single-bounce Physical Optics integral kernel
// Computes monostatic RCS contribution from first-hit ray data
__global__ void integral_PO(
    vec3* hit_pos,          // Input: hit positions
    vec3* hit_normal,       // Input: hit normals
    float* hit_dist,        // Input: hit distances
    int* hit_flag,          // Input: hit flags
    int n,                  // Total number of rays
    float k,                // Wave number (2π/λ)
    vec3 k_inc,             // Incident wave direction
    float ray_area,         // Area per ray
    cuFloatComplex* accum   // Output: accumulated complex field
);

// Multi-bounce Physical Optics integral kernel
// Computes monostatic RCS with multiple reflections and attenuation
__global__ void integral_PO_multibounce(
    vec3* hit_pos,              // Input: first hit positions
    vec3* hit_normal,           // Input: first hit normals
    float* hit_dist,            // Input: cumulative distances
    int* hit_count,             // Input: bounce counts per ray
    int n,                      // Total number of rays
    float k,                    // Wave number (2π/λ)
    vec3 k_inc,                 // Incident wave direction
    float ray_area,             // Area per ray
    cuFloatComplex* accum,      // Output: accumulated complex field
    float reflection_const      // Reflection coefficient per bounce
);

#endif // PO_KERNELS_CUH
