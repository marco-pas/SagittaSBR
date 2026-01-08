/*
 * Implementation of ray tracing kernels.
 */

#include "ray_kernels.cuh"
#include "ray.h"
#include <cfloat>

__global__ void launcher(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_flag, 
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


__global__ void launcher_multibounce(vec3* hit_pos, vec3* hit_normal, float* hit_dist, 
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

            if (k == 0) {
                hit_pos[idx] = rec.p;
                hit_normal[idx] = rec.normal;
            }

            hit_dist[idx] += rec.t;

            // Reflect
            current_dir = unit_vector(
                current_dir - 2.0f * dot(current_dir, rec.normal) * rec.normal
            );

            // Offset origin to avoid self-intersection
            current_origin = rec.p + rec.normal * EPS; // EPS should avoid self intersection

            // Optional grazing-angle escape
            if (fabsf(dot(current_dir, rec.normal)) < 1e-5f)
                break;
        }
        else {
            break;
        }
    }
}
