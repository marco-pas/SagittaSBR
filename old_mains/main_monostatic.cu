#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <string>
#include <iomanip>

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

// --- FANCY PRINTING HELPERS ---

void printGPUInfo() {
    cudaDeviceProp prop;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) return;
    cudaGetDeviceProperties(&prop, 0);
    std::cerr << "╔═══════════════════════════════════════════════════╗\n";
    std::cerr << "║                   GPU INFORMATION                 ║\n";
    std::cerr << "╠═══════════════════════════════════════════════════╣\n";
    std::cerr << "║ Device: " << prop.name << "\n";
    std::cerr << "║ Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB\n";
    std::cerr << "╚═══════════════════════════════════════════════════╝\n\n";
}

void printSeparator(const std::string& title = "") {
    std::cerr << "\n┌── " << std::left << std::setw(46) << title << " ┐\n";
}

void printEndSeparator() {
    std::cerr << "└──────────────────────────────────────────────────┘\n";
}

template<typename T>
void printKV(const std::string& key, const T& value, const std::string& unit = "") {
    std::cerr << "│ " << std::left << std::setw(20) << key << ": " << value << " " << unit << "\n";
}

std::string formatTime(double seconds) {
    if (seconds < 0.001) return std::to_string(seconds * 1e6) + " µs";
    if (seconds < 1.0) return std::to_string(seconds * 1000) + " ms";
    return std::to_string(seconds) + " s";
}

// --- KERNELS ---

__global__ void launcher(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_flag, int nx, int ny, vec3 llc, vec3 horiz, vec3 vert, vec3 ray_dir, hitable** world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float u = (i + 0.5f) / float(nx);
    float v = (j + 0.5f) / float(ny);
    vec3 ray_start = llc + u * horiz + v * vert;
    ray r(ray_start, ray_dir);
    hit_record rec;
    if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
        hit_flag[idx] = 1; hit_pos[idx] = rec.p; hit_normal[idx] = rec.normal; hit_dist[idx] = rec.t;
    } else { hit_flag[idx] = 0; }
}

__global__ void integral_PO(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_flag, int n, float k, vec3 k_inc, float ray_area, cuFloatComplex* accum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !hit_flag[idx]) return;
    if (dot(hit_normal[idx], -k_inc) <= 0.0f) return;
    float phase = 2.0f * k * hit_dist[idx];
    cuFloatComplex e = make_cuFloatComplex(cosf(phase), sinf(phase));
    cuFloatComplex factor = make_cuFloatComplex(0.0f, k * ray_area / (4.0f * M_PI));
    cuFloatComplex contrib = cuCmulf(e, factor);
    atomicAdd(&accum->x, contrib.x);
    atomicAdd(&accum->y, contrib.y);
}

__global__ void create_world(hitable** d_list, hitable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0.0f, 0.0f, 0.0f), 1.0f);
        *d_world  = new hitable_list(d_list, 1);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world) {
    delete d_list[0]; delete *d_world;
}

// --- MAIN ---

int main() {
    std::cerr << "╔═══════════════════════════════════════════════════╗\n";
    std::cerr << "║      MONOSTATIC RCS SWEEP - PHYSICAL OPTICS       ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════╝\n";
    
    printGPUInfo();

    auto cfg = loadConfig("config.txt");

    float phi_start = cfg.count("phi_start") ? cfg["phi_start"] : 0.0f;
    float phi_end   = cfg.count("phi_end")   ? cfg["phi_end"]   : 180.0f;
    int phi_samples = cfg.count("phi_samples") ? (int)cfg["phi_samples"] : 181;
    float theta_start = cfg.count("theta_start") ? cfg["theta_start"] : 90.0f;
    float theta_end   = cfg.count("theta_end")   ? cfg["theta_end"]   : 90.0f;
    int theta_samples = cfg.count("theta_samples") ? (int)cfg["theta_samples"] : 1;
    float freq      = cfg.count("freq")      ? cfg["freq"]      : 10.0e9f;
    float grid_size = cfg.count("grid_size") ? cfg["grid_size"] : 3.0f;
    int nx          = cfg.count("nx")        ? (int)cfg["nx"]   : 400;
    int ny          = cfg.count("ny")        ? (int)cfg["ny"]   : 400;

    const float c0 = 299792458.0f;
    const float lambda = c0 / freq;
    const float k = 2.0f * M_PI / lambda;
    const float ray_area = (grid_size * grid_size) / (nx * ny);
    const int N = nx * ny;
    const float distance_offset = 10.0f;

    printSeparator("SIMULATION PARAMETERS");
    printKV("Frequency", freq / 1e9, "GHz");
    printKV("Wavelength", lambda * 1000, "mm");
    printKV("Grid Size", std::to_string(nx) + "x" + std::to_string(ny));
    printKV("Total Rays", N);
    printKV("Phi Range", std::to_string(phi_start) + " to " + std::to_string(phi_end));
    printEndSeparator();

    printSeparator("MEMORY ALLOCATION");
    vec3 *hit_pos, *hit_normal; float *hit_dist; int *hit_flag; cuFloatComplex *accum;
    checkCudaErrors(cudaMallocManaged(&hit_pos, N * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&hit_normal, N * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&hit_dist, N * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&hit_flag, N * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&accum, sizeof(cuFloatComplex)));
    hitable **d_list, **d_world;
    checkCudaErrors(cudaMalloc(&d_list, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hitable*)));
    create_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "│  Managed memory and world setup complete.\n";
    printEndSeparator();

    std::ofstream outFile("rcs_results.csv");
    outFile << "# Frequency: " << freq << "\n# Grid: " << nx << "x" << ny << "\n";
    outFile << "# GridSize: " << grid_size << "\n# ThetaSamples: " << theta_samples << "\n";
    outFile << "# PhiSamples: " << phi_samples << "\n" << "theta,phi,rcs_m2,rcs_dbsm\n";

    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

    printSeparator("EXECUTING SWEEP");
    clock_t total_sweep_start = clock();
    int total_iterations = 0;

    for (int t = 0; t < theta_samples; ++t) {
        float theta_deg = theta_start + (theta_samples > 1 ? t * (theta_end - theta_start) / (theta_samples - 1) : 0);
        float theta_rad = theta_deg * M_PI / 180.0f;

        for (int p = 0; p < phi_samples; ++p) {
            clock_t iter_start = clock();
            
            float phi_deg = phi_start + (phi_samples > 1 ? p * (phi_end - phi_start) / (phi_samples - 1) : 0);
            float phi_rad = phi_deg * M_PI / 180.0f;

            vec3 dir(sinf(theta_rad)*cosf(phi_rad), sinf(theta_rad)*sinf(phi_rad), cosf(theta_rad));
            vec3 ray_dir = -dir;

            vec3 up(0, 1, 0);
            if (fabs(dir.y()) > 0.9f) up = vec3(1, 0, 0);
            vec3 u_vec = unit_vector(cross(up, dir)) * grid_size;
            vec3 v_vec = unit_vector(cross(dir, u_vec)) * grid_size;
            vec3 llc = (dir * distance_offset) - 0.5f * u_vec - 0.5f * v_vec;

            accum->x = 0.0f; accum->y = 0.0f;

            launcher<<<blocks, threads>>>(hit_pos, hit_normal, hit_dist, hit_flag, nx, ny, llc, u_vec, v_vec, ray_dir, d_world);
            cudaDeviceSynchronize();

            integral_PO<<<(N+255)/256, 256>>>(hit_pos, hit_normal, hit_dist, hit_flag, N, k, ray_dir, ray_area, accum);
            cudaDeviceSynchronize();

            float sigma = 4.0f * M_PI * (accum->x * accum->x + accum->y * accum->y);
            float rcs_dBsm = 10.0f * log10f(fmaxf(sigma, 1e-10f));

            outFile << theta_deg << "," << phi_deg << "," << sigma << "," << rcs_dBsm << "\n";
            
            clock_t iter_end = clock();
            double iter_time = double(iter_end - iter_start) / CLOCKS_PER_SEC;
            
            std::cerr << "│  Point " << std::setw(3) << total_iterations + 1 
                      << " | Phi: " << std::setw(6) << phi_deg 
                      << " | Theta: " << std::setw(6) << theta_deg 
                      << " | Time: " << formatTime(iter_time) << "\n";
            
            total_iterations++;
        }
    }
    
    clock_t total_sweep_end = clock();
    double total_time = double(total_sweep_end - total_sweep_start) / CLOCKS_PER_SEC;
    printEndSeparator();

    printSeparator("PERFORMANCE SUMMARY");
    printKV("Total Sweep Time", formatTime(total_time));
    printKV("Total Points", total_iterations);
    printKV("Average Time/Point", formatTime(total_time / total_iterations));
    printEndSeparator();

    printSeparator("CLEANUP");
    free_world<<<1,1>>>(d_list, d_world);
    cudaDeviceSynchronize();
    cudaFree(hit_pos); cudaFree(hit_normal); cudaFree(hit_dist);
    cudaFree(hit_flag); cudaFree(accum); cudaFree(d_list); cudaFree(d_world);
    std::cerr << "│  Device memory released. Success.\n";
    printEndSeparator();

    return 0;
}