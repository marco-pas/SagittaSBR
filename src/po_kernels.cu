/*
 * Implementation of Physical Optics integration kernels.
 */

#include "po_kernels.cuh"
#include <cmath>

__global__ void integral_PO(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_flag, 
                            int n, float k, vec3 k_inc, float ray_area, cuFloatComplex* accum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n || !hit_flag[idx]) return;

    // 1. Obliquity factor: In monostatic backscatter, this is dot(n, -k_inc)
    float cos_theta = dot(hit_normal[idx], -k_inc);
    if (cos_theta <= 0.0f) return;

    // 2. Monostatic Phase
    float phase = 2.0f * k * hit_dist[idx];
    cuFloatComplex e = make_cuFloatComplex(cosf(phase), sinf(phase));

    // 3. THE FIX: Multiply by 2.0f because J = 2 * (n x H)
    // The standard PO factor is (j * k / (4 * PI)) * 2.0
    float mag = (k * ray_area / (4.0f * M_PI)) * 2.0f;
    
    cuFloatComplex contrib = make_cuFloatComplex(-mag * sinf(phase), mag * cosf(phase)); 
    // representing j * factor * exp(j*phase)

    atomicAdd(&accum->x, contrib.x);    // real part 
    atomicAdd(&accum->y, contrib.y);    // imaginary part
}

__global__ void integral_PO_multibounce(vec3* hit_pos, vec3* hit_normal, float* hit_dist, 
                                       int* hit_count, int n, float k, vec3 k_inc, 
                                       float ray_area, cuFloatComplex* accum, 
                                       float reflection_const) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n || hit_count[idx] == 0) return;

    float total_refl_coeff = powf(reflection_const, (float)hit_count[idx]);

    // 1. Obliquity factor: In monostatic backscatter, this is dot(n, -k_inc)
    float cos_theta = dot(hit_normal[idx], -k_inc);
    if (cos_theta <= 0.0f) return;

    // 2. Monostatic Phase
    float phase = 2.0f * k * hit_dist[idx];
    cuFloatComplex e = make_cuFloatComplex(cosf(phase), sinf(phase));

    // 3. THE FIX: Multiply by 2.0f because J = 2 * (n x H)
    // The standard PO factor is (j * k / (4 * PI)) * 2.0
    float mag = (k * ray_area / (4.0f * M_PI)) * 2.0f * total_refl_coeff;
    
    cuFloatComplex contrib = make_cuFloatComplex(-mag * sinf(phase), mag * cosf(phase)); 
    // representing j * factor * exp(j*phase)

    atomicAdd(&accum->x, contrib.x);    // real part 
    atomicAdd(&accum->y, contrib.y);    // imaginary part
}
