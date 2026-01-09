/*
 * Implementation of ray tracing kernels.
 */

#include "ray_kernels.cuh"
#include "ray.h"
#include <cfloat>

__global__ void launcher_singlebounce(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_flag, 
                         int nx, int ny, vec3 llc, vec3 horiz, vec3 vert, vec3 ray_dir, 
                         hitable** world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return; // check we in the range
    int idx = j * nx + i;

    float u = (i + 0.5f) / float(nx);
    float v = (j + 0.5f) / float(ny);

    vec3 ray_start = llc + u * horiz + v * vert; // start the ray launch

    ray r(ray_start, ray_dir);

    // now here we still only allow for 1 hit
    hit_record rec;
    if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
        hit_flag[idx] = 1; 
        hit_pos[idx] = rec.p; 
        hit_normal[idx] = rec.normal; 
        hit_dist[idx] = rec.t;
    } else { 
        hit_flag[idx] = 0; 
    }
}

// OLD kernel, before 

// __global__ void launcher_multibounce(vec3* hit_pos, vec3* hit_normal, float* hit_dist, 
//                                      int* hit_count, int nx, int ny, vec3 llc, vec3 horiz, 
//                                      vec3 vert, vec3 ray_dir, hitable** world, int max_bounces) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i >= nx || j >= ny) return;
//     int idx = j * nx + i;

//     float u = (i + 0.5f) / float(nx);
//     float v = (j + 0.5f) / float(ny);

//     vec3 current_origin = llc + u * horiz + v * vert;
//     vec3 current_dir = ray_dir; 
    
//     hit_dist[idx] = 0.0f;
//     hit_count[idx] = 0;

//     for (int k = 0; k < max_bounces; k++) {
//         ray r(current_origin, current_dir);
//         hit_record rec;

//         if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
//             hit_count[idx]++;
            
//             // Store the very first hit position/normal for the buffer
//             if (k == 0) {
//                 hit_pos[idx] = rec.p;
//                 hit_normal[idx] = rec.normal;
//             }

//             hit_dist[idx] += rec.t;

//             // Update for next bounce - simple reflection
//             current_origin = rec.p; 
//             current_dir = current_dir - 2.0f * dot(current_dir, rec.normal) * rec.normal;

//         } else {
//             break; // Ray missed everything, stop bouncing
//         }
//     }
// }


// __global__ void launcher(vec3* hit_pos, vec3* hit_normal, float* hit_dist, 
//                                      int* hit_count, int nx, int ny, vec3 llc, vec3 horiz, 
//                                      vec3 vert, vec3 ray_dir, hitable** world, int max_bounces) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i >= nx || j >= ny) return;
//     int idx = j * nx + i;

//     float u = (i + 0.5f) / float(nx);
//     float v = (j + 0.5f) / float(ny);

//     vec3 current_origin = llc + u * horiz + v * vert;
//     vec3 current_dir = unit_vector(ray_dir);

//     hit_dist[idx] = 0.0f;
//     hit_count[idx] = 0;

//     const float EPS = 1e-4f;

//     for (int k = 0; k < max_bounces; k++) {
//         ray r(current_origin, current_dir);
//         hit_record rec;

//         if ((*world)->hit(r, EPS, FLT_MAX, rec)) {
//             hit_count[idx]++;

//             if (k == 0) {
//                 hit_pos[idx] = rec.p;
//                 hit_normal[idx] = rec.normal;
//             }

//             hit_dist[idx] += rec.t;

//             // Reflect
//             current_dir = unit_vector(
//                 current_dir - 2.0f * dot(current_dir, rec.normal) * rec.normal
//             );

//             // Offset origin to avoid self-intersection
//             current_origin = rec.p + rec.normal * EPS; // EPS should avoid self intersection

//             // Optional grazing-angle escape
//             if (fabsf(dot(current_dir, rec.normal)) < 1e-5f)
//                 break;
//         }
//         else {
//             break;
//         }
//     }
// }


__global__ void launcher(vec3* hit_pos, vec3* hit_normal, float* hit_dist, 
                         int* hit_count, int nx, int ny, vec3 llc, vec3 horiz, 
                         vec3 vert, vec3 ray_dir, hitable** world, int max_bounces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;

    float u = (i + 0.5f) / float(nx);
    float v = (j + 0.5f) / float(ny);

    vec3 current_origin = llc + u * horiz + v * vert;
    vec3 current_dir = unit_vector(ray_dir);

    hit_dist[idx] = 0.0f;
    hit_count[idx] = 0;

    const float EPS = 1e-4f;

    for (int k = 0; k < max_bounces; k++) {
        ray r(current_origin, current_dir);
        hit_record rec;

        if ((*world)->hit(r, EPS, FLT_MAX, rec)) {
            hit_count[idx]++;
            hit_dist[idx] += rec.t;

            // Store LAST hit position and normal (overwrite each time)
            hit_pos[idx] = rec.p;
            hit_normal[idx] = rec.normal;

            // Reflect
            current_dir = unit_vector(
                current_dir - 2.0f * dot(current_dir, rec.normal) * rec.normal
            );

            // Offset origin to avoid self-intersection
            current_origin = rec.p + rec.normal * EPS;

            // Optional grazing-angle escape
            if (fabsf(dot(current_dir, rec.normal)) < 1e-5f)
                break;
        }
        else {
            break;
        }
    }
}


__global__ void launcher_vector(
    vec3* hit_pos, vec3* hit_normal, vec3* hit_dir, vec3* hit_pol,
    float* hit_dist, int* hit_count, 
    int nx, int ny, vec3 llc, vec3 u, vec3 v, vec3 ray_dir, 
    hitable **world, int max_bounces
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= nx) || (j >= ny)) return;

    int pixel_index = j * nx + i;

    // Initial ray setup
    float u_coord = float(i) / float(nx);
    float v_coord = float(j) / float(ny);
    ray r(llc + u_coord * u + v_coord * v, ray_dir);

    // Initial Polarization: 
    // Usually assumed to be 'Vertical' or 'Horizontal' relative to initial grid
    // Here we initialize it to match the 'v' vector of the grid
    vec3 current_pol = unit_vector(v); 
    
    float total_dist = 0;
    int bounces = 0;
    hit_record rec;

    // Trace the ray through multiple bounces
    while (bounces < max_bounces && (*world)->hit(r, 0.001f, 1e20f, rec)) {
        bounces++;
        total_dist += rec.t;

        // 1. Reflect the Ray Direction (Standard Snell's Law)
        // v_refl = v_inc - 2 * dot(v_inc, n) * n
        vec3 reflected_direction = unit_vector(r.direction() - 2.0f * dot(r.direction(), rec.normal) * rec.normal);

        // 2. Reflect the Polarization Vector (PEC Boundary Condition)
        // E_refl = -E_inc + 2 * dot(E_inc, n) * n
        // This ensures the tangential E-field is zero at the surface
        current_pol = unit_vector(-current_pol + 2.0f * dot(current_pol, rec.normal) * rec.normal);

        // Update ray for next bounce
        r = ray(rec.p, reflected_direction);
    }

    // Save final state to buffers for the PO integral
    hit_count[pixel_index]  = bounces;
    if (bounces > 0) {
        hit_pos[pixel_index]    = rec.p;
        hit_normal[pixel_index] = rec.normal;
        hit_dir[pixel_index]    = r.direction(); // The exit direction
        hit_pol[pixel_index]    = current_pol;   // The final polarization
        hit_dist[pixel_index]   = total_dist;
    }
}