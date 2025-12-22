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

// Function to print GPU information
void printGPUInfo() {
    cudaDeviceProp prop;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "✗ No CUDA-capable devices found\n";
        return;
    }
    
    cudaGetDeviceProperties(&prop, 0);
    
    std::cerr << "╔═══════════════════════════════════════════════════╗\n";
    std::cerr << "║                   GPU INFORMATION                 ║\n";
    std::cerr << "╠═══════════════════════════════════════════════════╣\n";
    std::cerr << "║ Device: " << prop.name << "\n";
    std::cerr << "║ Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cerr << "║ Total Global Memory: " 
              << prop.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB\n";
    std::cerr << "║ Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cerr << "║ Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cerr << "╚═══════════════════════════════════════════════════╝\n";
    std::cerr << "\n";
}

// Function to print a separator line
void printSeparator(const std::string& title = "") {
    std::cerr << "\n";
    if (!title.empty()) {
        std::cerr << "┌── " << title << " ";
        int remaining = 50 - title.length() - 4;
        for (int i = 0; i < remaining; i++) std::cerr << "─";
        std::cerr << "┐\n";
    } else {
        std::cerr << "┌───────────────────────────────────────────────────┐\n";
    }
}

void printEndSeparator() {
    std::cerr << "└──────────────────────────────────────────────────┘\n";
}

// Function to print key-value pair
template<typename T>
void printKV(const std::string& key, const T& value, const std::string& unit = "") {
    std::cerr << "│ " << key << ": " << value;
    if (!unit.empty()) std::cerr << " " << unit;
    std::cerr << "\n";
}

// Function to format time
std::string formatTime(double seconds) {
    if (seconds < 0.001) {
        return std::to_string(seconds * 1e6) + " µs";
    } else if (seconds < 1.0) {
        return std::to_string(seconds * 1000) + " ms";
    } else {
        return std::to_string(seconds) + " s";
    }
}

// MAIN
int main() {
    // Print program header
    std::cerr << "\n";
    std::cerr << "╔═══════════════════════════════════════════════════╗\n";
    std::cerr << "║      PHYSICAL OPTICS RCS SIMULATION - CUDA        ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════╝\n";
    
    // Print GPU information
    printGPUInfo();
    
    // Parameters
    const int nx = 600;
    const int ny = 400;
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

    vec3 llc(-H / 2.0f, -V / 2.0f, 4.0f);
    vec3 horiz(H, 0.0f, 0.0f);
    vec3 vert(0.0f, V, 0.0f);
    vec3 origin(0.0f, 0.0f, 0.0f);

    const float ray_area = (H * V) / (nx * ny);
    const int N = nx * ny;

    // Print simulation parameters
    printSeparator("SIMULATION PARAMETERS");
    printKV(" Ray Grid", std::to_string(nx) + " × " + std::to_string(ny));
    printKV(" Block Size", std::to_string(tx) + " × " + std::to_string(ty));
    printKV(" Frequency", freq / 1e9, "GHz");
    printKV(" Wavelength", lambda * 1000, "mm");
    printKV(" Wave Number k", k, "rad/m");
    printKV(" Illumination Area", H * V, "m²");
    printKV(" Total Rays", N, "");
    printKV(" Ray Density", N / (H * V), "rays/m²");
    printKV(" Ray Area", ray_area, "m²");
    printEndSeparator();

    // Allocate buffers
    printSeparator("MEMORY ALLOCATION");
    std::cerr << "│  Allocating device memory...\n";
    
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

    std::cerr << "│  Memory allocated successfully\n";
    printKV(" Total Allocated", (N * (sizeof(vec3)*2 + sizeof(float) + sizeof(int)) + 
                                sizeof(cuFloatComplex) + 2*sizeof(hitable*)) / (1024*1024.0), "MB");
    printEndSeparator();

    // Create world
    printSeparator("SCENE SETUP");
    std::cerr << "│  Creating sphere object...\n";
    create_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "│  Sphere created (center: [0,0,-4], radius: 2)\n";
    printEndSeparator();

    // Launch rays
    dim3 threads(tx, ty);
    dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);

    printSeparator("RAY TRACING");
    std::cerr << "│  Launching " << blocks.x << " × " << blocks.y 
              << " blocks with " << threads.x << " × " << threads.y << " threads\n";
    
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
    
    std::cerr << "│  Ray tracing completed\n";
    printKV(" Time", formatTime(timer_seconds));

    // Count number of hits
    int hit_count = 0;
    for (int i = 0; i < nx * ny; ++i) {
        hit_count += hit_flag[i];
    }
    
    float hit_percentage = (hit_count * 100.0f) / N;
    printKV(" Rays Hit Object", hit_count);
    printKV(" Hit Percentage", std::to_string(hit_percentage).substr(0, 5) + "%");
    printEndSeparator();

    // PO Integral
    printSeparator("PO INTEGRAL");
    std::cerr << "│  Calculating Physical Optics integral...\n";
    
    int threads1D = 256;
    int blocks1D = (N + threads1D - 1) / threads1D;
    
    std::cerr << "│  Using " << blocks1D << " blocks with " << threads1D << " threads\n";
    
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
    
    std::cerr << "│  PO integral calculated\n";
    printKV(" Time", formatTime(timer_seconds));
    printEndSeparator();

    // RCS
    printSeparator("RCS CALCULATION");
    float sigma = 4.0f * M_PI * (accum->x * accum->x + accum->y * accum->y);
    
    std::cerr << "│  Accumulated Field: (" << accum->x << ", " << accum->y << "i)\n";
    printKV(" RCS Value", sigma, "m²");
    
    // Convert to dBsm
    float rcs_dBsm = 10.0f * log10f(sigma / 1.0f);
    printKV(" RCS", rcs_dBsm, "dBsm");
    printEndSeparator();

    // Cleanup
    printSeparator("CLEANUP");
    std::cerr << "│  Freeing device memory...\n";
    
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
    
    std::cerr << "│  Memory freed successfully\n";
    printEndSeparator();


    return 0;
}