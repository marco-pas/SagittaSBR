#include <iostream>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"

// CUDA error checking
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line
                  << " '" << func << "'\n";
        cudaDeviceReset();
        exit(1);
    }
}

__global__ void launcher(
    vec3* hit_pos,
    vec3* hit_normal,
    float* hit_dist,
    int* hit_flag,
    int nx, int ny,
    vec3 llc,
    vec3 horiz,
    vec3 vert,
    hitable** world
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;

    // grid coordinates
    float u = (i + 0.5f) / float(nx);
    float v = (j + 0.5f) / float(ny);

    // ray starting position on the launch plane
    vec3 ray_start = llc + u * horiz + v * vert;

    // all rays launch along -z
    vec3 ray_dir = vec3(0.0f, 0.0f, -1.0f);

    ray r(ray_start, ray_dir);

    hit_record rec;
    if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
        hit_flag[idx]   = 1;
        hit_pos[idx]    = rec.p;
        hit_normal[idx] = rec.normal;
        hit_dist[idx]   = rec.t;
    } else {
        hit_flag[idx] = 0;
    }
}

// PO integral kernel (scalar, single-pol, monostatic)
__global__ void integral_PO(
    vec3* hit_pos,
    vec3* hit_normal,
    float* hit_dist,
    int* hit_flag,
    int n,
    float k,
    vec3 k_inc,
    float ray_area,
    cuFloatComplex* accum
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !hit_flag[idx]) return;

    // Shadowing (illuminated region only)
    if (dot(hit_normal[idx], -k_inc) <= 0.0f) return;

    float phase =
        2.0f * k * hit_dist[idx]
        - dot(k_inc, hit_pos[idx]);

    cuFloatComplex e =
        make_cuFloatComplex(cosf(phase), sinf(phase));

    cuFloatComplex factor =
        make_cuFloatComplex(0.0f, k * ray_area / (4.0f * M_PI));

    cuFloatComplex contrib = cuCmulf(e, factor);

    atomicAdd(&accum->x, contrib.x);
    atomicAdd(&accum->y, contrib.y);
}

// World creation / destruction
__global__ void create_world(hitable** d_list, hitable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Sphere centered in domain
        d_list[0] = new sphere(vec3(0.0f, 0.0f, -4.0f), 2.0f);
        *d_world  = new hitable_list(d_list, 1);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world) {
    delete d_list[0];
    delete *d_world;
}

// MAIN
int main() {

    // Parameters
    const int nx = 600;
    const int ny =400;
    const int tx = 8;
    const int ty = 8;

    const float freq = 10.0e9f;
    const float c0   = 299792458.0f;
    const float lambda = c0 / freq;
    const float k = 2.0f * M_PI / lambda;

    const float phi   = 0.3f;
    const float theta = 0.2f;

    vec3 k_inc(
        sinf(theta) * cosf(phi),
        sinf(theta) * sinf(phi),
        cosf(theta)
    );

    const float H = 4.0f;
    const float V = 2.0f;

    vec3 llc(-H / 2.0f, -V / 2.0f, 0.0f);
    vec3 horiz(H, 0.0f, 0.0f);
    vec3 vert(0.0f, V, 0.0f);
    vec3 origin(0.0f, 0.0f, 0.0f);

    const float ray_area = (H * V) / (nx * ny);
    const int N = nx * ny;

    // Set-up prints
    std::cerr << "--- Calculating RCS for a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";
    std::cerr << "Frequency set to " << freq << " Hz.\n";
    std::cerr << "There are " << N
              << " rays. Density is "
              << (N / (V * H))
              << " (rays / m^2).\n";

    // Allocate buffers
    vec3* hit_pos;
    vec3* hit_normal;
    float* hit_dist;
    int* hit_flag;

    checkCudaErrors(cudaMallocManaged(&hit_pos,    N * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&hit_normal, N * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&hit_dist,   N * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&hit_flag,   N * sizeof(int)));

    cuFloatComplex* accum;
    checkCudaErrors(cudaMallocManaged(&accum, sizeof(cuFloatComplex)));
    accum->x = accum->y = 0.0f;

    // World
    hitable** d_list;
    hitable** d_world;
    checkCudaErrors(cudaMalloc(&d_list, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hitable*)));

    create_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaDeviceSynchronize());

    // Launch rays
    dim3 threads(tx, ty);
    dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);

    std::cerr << "--- Launching Rays!\n";
    clock_t start = clock();

    launcher<<<blocks, threads>>>(
        hit_pos,
        hit_normal,
        hit_dist,
        hit_flag,
        nx, ny,
        llc,
        horiz,
        vert,
        d_world
    );
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t stop = clock();
    double timer_seconds = double(stop - start) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds
              << " seconds for the collisions.\n";

    // Count number of hits
    int hit_count = 0;
    for (int i = 0; i < nx * ny; ++i) {
        hit_count += hit_flag[i];
}

std::cerr << "Number of rays hitting object: "
          << hit_count << "\n";


    // PO Integral
    std::cerr << "--- Calculating Integral!\n";

    int threads1D = 256;
    int blocks1D = (N + threads1D - 1) / threads1D;

    start = clock();

    integral_PO<<<blocks1D, threads1D>>>(
        hit_pos,
        hit_normal,
        hit_dist,
        hit_flag,
        N,
        k,
        k_inc,
        ray_area,
        accum
    );
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    timer_seconds = double(stop - start) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds
              << " seconds for the integral calculation.\n";

    // RCS
    float sigma =
        4.0f * M_PI *
        (accum->x * accum->x + accum->y * accum->y);

    std::cerr << "RCS value is " << sigma << " m^2\n";

    // Cleanup
    std::cerr << "--- Cleaning up!\n";

    free_world<<<1,1>>>(d_list, d_world);
    cudaDeviceSynchronize();

    cudaFree(hit_pos);
    cudaFree(hit_normal);
    cudaFree(hit_dist);
    cudaFree(hit_flag);
    cudaFree(accum);
    cudaFree(d_list);
    cudaFree(d_world);

    cudaDeviceReset();
    return 0;
}
