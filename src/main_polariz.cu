// At the moment this is pretty wrong!

// Try to fix !!!!!!!!!!!!!!!!!!

#include <iostream>       // Console I/O
#include <fstream>        // File I/O
#include <cmath>          // Math
#include <ctime>          // Timing
#include <string>         // Strings
#include <iomanip>        // Output formatting
#include <cstdlib>        // Standard lib
#include <sys/stat.h>     // Directory operations
#include <errno.h>        // Errors

// CUDA libraries
#include <cuda_runtime.h> // CUDA runtime API
#include <cuComplex.h>    // Complex types

// Project-specific headers
#include "vec3.h"           // 3D vectors
#include "hitable_list.h"   // List of hitable objects
#include "cuda_utils.h"     // Error checking
#include "printing_utils.h" // Prints
#include "config_parser.h"  // File parsing
#include "ray_kernels.cuh"  // Ray tracing kernels
#include "po_kernels.cuh"   // Physical Optics integration kernels
#include "world_setup.cuh"  // Scene/world setup

// Kernel to compute hit statistics on device
__global__ void compute_hit_stats(int *hit_count, int N, int *max_hits, int *zero_count, long long *sum_hits, int *non_zero_count) {
    // Use shared memory for reduction
    __shared__ int s_max[256];
    __shared__ int s_zero[256];
    __shared__ long long s_sum[256];
    __shared__ int s_nonzero[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    s_max[tid] = 0;
    s_zero[tid] = 0;
    s_sum[tid] = 0;
    s_nonzero[tid] = 0;
    
    // Process elements
    if (idx < N) {
        int val = hit_count[idx];
        if (val == 0) {
            s_zero[tid] = 1;
        } else {
            s_max[tid] = val;
            s_sum[tid] = val;
            s_nonzero[tid] = 1;
        }
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = max(s_max[tid], s_max[tid + s]);
            s_zero[tid] += s_zero[tid + s];
            s_sum[tid] += s_sum[tid + s];
            s_nonzero[tid] += s_nonzero[tid + s];
        }
        __syncthreads();
    }
    
    // Write block results to global memory
    if (tid == 0) {
        atomicMax(max_hits, s_max[0]);
        atomicAdd(zero_count, s_zero[0]);
        atomicAdd((unsigned long long*)sum_hits, (unsigned long long)s_sum[0]);
        atomicAdd(non_zero_count, s_nonzero[0]);
    }
}

// Create directory if it doesn't exist
bool createDirectory(const std::string& path) {
    #ifdef _WIN32
        if (_mkdir(path.c_str()) == 0) {
            return true; // created
        }
    #else
        if (mkdir(path.c_str(), 0777) == 0) {
            return true; // created
        }
    #endif
    return errno == EEXIST;
}

int main(int argc, char** argv) {
    std::cerr << "╔════════════════════════════════════════════════════╗\n";
    std::cerr << "║  SagittaSBR  >>---->  VECTOR POLARIZATION RCS      ║\n";
    std::cerr << "╚════════════════════════════════════════════════════╝\n";

    // Load configuration
    auto cfg = loadConfig("config.txt");

    // Parse configuration parameters
    float phi_start = cfg.count("phi_start") ? cfg["phi_start"] : 0.0f;
    float phi_end   = cfg.count("phi_end")   ? cfg["phi_end"]   : 180.0f;
    int phi_samples = cfg.count("phi_samples") ? (int)cfg["phi_samples"] : 181;
    float theta_start = cfg.count("theta_start") ? cfg["theta_start"] : 90.0f;
    float theta_end   = cfg.count("theta_end")   ? cfg["theta_end"]   : 90.0f;
    int theta_samples = cfg.count("theta_samples") ? (int)cfg["theta_samples"] : 1;
    float grid_size = cfg.count("grid_size") ? cfg["grid_size"] : 3.0f;
    int nx          = cfg.count("nx")        ? (int)cfg["nx"]   : 5000;
    int ny          = cfg.count("ny")        ? (int)cfg["ny"]   : 5000;
    int tpbx        = cfg.count("tpbx")        ? (int)cfg["tpbx"]   : 32;
    int tpby        = cfg.count("tpby")        ? (int)cfg["tpby"]   : 8;
    int max_bounces = cfg.count("max_bounces")    ? (int)cfg["max_bounces"]   : 20;
    float reflection_const = cfg.count("reflection_const")    ? cfg["reflection_const"] : 1.0f;

    bool showInfoGPU = cfg.count("showInfoGPU")    ? (bool)cfg["showInfoGPU"] : false;
    bool showHitStats = cfg.count("showHitStats")    ? (bool)cfg["showHitStats"] : false;
    
    float freq;
    if (argc > 1) {
        freq = std::atof(argv[1]);
        std::cout << "Using frequency from command line: " << freq << " Hz\n";
    } else if (cfg.count("freq")) {
        freq = cfg["freq"];
        std::cout << "Using frequency from config file: " << freq << " Hz\n";
    } else {
        freq = 10.0e9f;
        std::cout << "Using default frequency: " << freq << " Hz\n";
    }

    // Physical constants
    const float c0 = 299792458.0f;
    const float lambda = c0 / freq;
    const float k = 2.0f * M_PI / lambda;
    const float ray_area = (grid_size * grid_size) / (nx * ny);
    const int N = nx * ny;
    const float distance_offset = 10.0f;

    if (showInfoGPU) printGPUInfo();

    // Print simulation parameters
    printSeparator("SIMULATION PARAMETERS");
    printKV("Frequency", freq / 1e9, "GHz");
    printKV("Wavelength", lambda * 1000, "mm");
    printKV("Wave Number k", k, "rad/m");
    printKV("Illumination Area", grid_size * grid_size, "m²");
    printKV("Grid Size", std::to_string(nx) + "x" + std::to_string(ny));
    printKV("Total Rays", N);
    printKV("Ray Density", N / (grid_size * grid_size), "rays/m²");
    printKV("Ray Area", ray_area, "m²");
    printKV("Phi Range", std::to_string(phi_start) + " to " + std::to_string(phi_end));
    printKV("Theta Range", std::to_string(theta_start) + " to " + std::to_string(theta_end));
    printEndSeparator();

    // Memory allocation
    printSeparator("MEMORY ALLOCATION");
    vec3 *d_hit_pos, *d_hit_normal, *d_hit_dir, *d_hit_pol; 
    float *d_hit_dist; 
    int *d_hit_count; 
    cuFloatComplex *d_accum_theta, *d_accum_phi;
    
    checkCudaErrors(cudaMalloc(&d_hit_pos, N * sizeof(vec3)));
    checkCudaErrors(cudaMalloc(&d_hit_normal, N * sizeof(vec3)));
    checkCudaErrors(cudaMalloc(&d_hit_dir, N * sizeof(vec3)));
    checkCudaErrors(cudaMalloc(&d_hit_pol, N * sizeof(vec3)));
    checkCudaErrors(cudaMalloc(&d_hit_dist, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_hit_count, N * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_accum_theta, sizeof(cuFloatComplex)));
    checkCudaErrors(cudaMalloc(&d_accum_phi, sizeof(cuFloatComplex)));

    cuFloatComplex *h_accum_theta, *h_accum_phi;
    checkCudaErrors(cudaMallocHost(&h_accum_theta, sizeof(cuFloatComplex)));
    checkCudaErrors(cudaMallocHost(&h_accum_phi, sizeof(cuFloatComplex)));

    int *d_max_hits, *d_zero_count, *d_non_zero_count;
    long long *d_sum_hits;
    checkCudaErrors(cudaMalloc(&d_max_hits, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_zero_count, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_sum_hits, sizeof(long long)));
    checkCudaErrors(cudaMalloc(&d_non_zero_count, sizeof(int)));

    int *h_max_hits, *h_zero_count, *h_non_zero_count;
    long long *h_sum_hits;
    checkCudaErrors(cudaMallocHost(&h_max_hits, sizeof(int)));
    checkCudaErrors(cudaMallocHost(&h_zero_count, sizeof(int)));
    checkCudaErrors(cudaMallocHost(&h_sum_hits, sizeof(long long)));
    checkCudaErrors(cudaMallocHost(&h_non_zero_count, sizeof(int)));

    cudaStream_t compute_stream;
    checkCudaErrors(cudaStreamCreate(&compute_stream));

    hitable **d_list, **d_world;
    checkCudaErrors(cudaMalloc(&d_list, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hitable*)));
    create_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "│  Device memory and world setup complete.\n";
    printEndSeparator();

    std::string outputDir = "output";
    createDirectory(outputDir);
    std::string outputFile = outputDir + "/rcs_results.csv";
    std::ofstream outFile(outputFile);
    
    outFile << "# Frequency: " << freq << " Hz\n";
    outFile << "# Wavelength: " << lambda << " m\n";
    outFile << "# Wave Number k: " << k << " rad/m\n";
    outFile << "# Grid: " << nx << "x" << ny << "\n";
    outFile << "# GridSize: " << grid_size << " m\n";
    outFile << "# RayArea: " << ray_area << " m²\n";
    outFile << "# ThetaSamples: " << theta_samples << "\n";
    outFile << "# PhiSamples: " << phi_samples << "\n";
    outFile << "# MaxBounces: " << max_bounces << "\n";
    outFile << "# ReflectionConst: " << reflection_const << "\n";
    outFile << "theta,phi,rcs_m2,rcs_dbsm\n";

    dim3 threads(tpbx, tpby);
    dim3 blocks((nx + tpbx - 1) / tpbx, (ny + tpby - 1) / tpby);
    int stats_threads = 256;
    int stats_blocks = (N + stats_threads - 1) / stats_threads;

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

            // Monostatic Direction
            vec3 obs_dir(sinf(theta_rad)*cosf(phi_rad), sinf(theta_rad)*sinf(phi_rad), cosf(theta_rad));
            vec3 ray_dir = -obs_dir;

            // Observer Polarization Basis
            vec3 world_up(0, 1, 0);
            if (fabs(obs_dir.y()) > 0.99f) world_up = vec3(1, 0, 0);
            vec3 obs_phi = unit_vector(cross(world_up, obs_dir));
            vec3 obs_theta = cross(obs_dir, obs_phi);

            // Ray Grid Setup
            vec3 u_vec = obs_phi * grid_size;
            vec3 v_vec = obs_theta * grid_size;
            vec3 llc = (obs_dir * distance_offset) - 0.5f * u_vec - 0.5f * v_vec;

            // Reset Async
            checkCudaErrors(cudaMemsetAsync(d_accum_theta, 0, sizeof(cuFloatComplex), compute_stream));
            checkCudaErrors(cudaMemsetAsync(d_accum_phi, 0, sizeof(cuFloatComplex), compute_stream));

            // 1. Launch Rays (Vector version)
            launcher_vector<<<blocks, threads, 0, compute_stream>>>(
                d_hit_pos, d_hit_normal, d_hit_dir, d_hit_pol, 
                d_hit_dist, d_hit_count, nx, ny, llc, u_vec, v_vec, 
                ray_dir, d_world, max_bounces
            );

            // 2. Vector PO Integral
            integral_PO_vector<<<stats_blocks, stats_threads, 0, compute_stream>>>(
                d_hit_pos, d_hit_normal, d_hit_dir, d_hit_pol, 
                d_hit_dist, d_hit_count, N, k, obs_dir, obs_theta, obs_phi, 
                ray_area, d_accum_theta, d_accum_phi, reflection_const
            );

            // 3. Stats and Data Transfer Async
            if (showHitStats) {
                checkCudaErrors(cudaMemsetAsync(d_max_hits, 0, sizeof(int), compute_stream));
                checkCudaErrors(cudaMemsetAsync(d_zero_count, 0, sizeof(int), compute_stream));
                checkCudaErrors(cudaMemsetAsync(d_sum_hits, 0, sizeof(long long), compute_stream));
                checkCudaErrors(cudaMemsetAsync(d_non_zero_count, 0, sizeof(int), compute_stream));
                compute_hit_stats<<<stats_blocks, stats_threads, 0, compute_stream>>>(d_hit_count, N, d_max_hits, d_zero_count, d_sum_hits, d_non_zero_count);
                checkCudaErrors(cudaMemcpyAsync(h_max_hits, d_max_hits, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
                checkCudaErrors(cudaMemcpyAsync(h_zero_count, d_zero_count, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
                checkCudaErrors(cudaMemcpyAsync(h_sum_hits, d_sum_hits, sizeof(long long), cudaMemcpyDeviceToHost, compute_stream));
                checkCudaErrors(cudaMemcpyAsync(h_non_zero_count, d_non_zero_count, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
            }
            checkCudaErrors(cudaMemcpyAsync(h_accum_theta, d_accum_theta, sizeof(cuFloatComplex), cudaMemcpyDeviceToHost, compute_stream));
            checkCudaErrors(cudaMemcpyAsync(h_accum_phi, d_accum_phi, sizeof(cuFloatComplex), cudaMemcpyDeviceToHost, compute_stream));

            // Sync Stream
            checkCudaErrors(cudaStreamSynchronize(compute_stream));

            // RCS Calculation: Sum of orthogonal polarization power
            float sigma = 4.0f * M_PI * (h_accum_theta->x * h_accum_theta->x + h_accum_theta->y * h_accum_theta->y +
                                         h_accum_phi->x * h_accum_phi->x + h_accum_phi->y * h_accum_phi->y);
            float rcs_dBsm = 10.0f * log10f(fmaxf(sigma, 1e-10f));

            outFile << theta_deg << "," << phi_deg << "," << sigma << "," << rcs_dBsm << "\n";
            
            clock_t iter_end = clock();
            double iter_time = double(iter_end - iter_start) / CLOCKS_PER_SEC;

            if (showHitStats) {
                float avg_hits = (*h_non_zero_count > 0) ? (float)(*h_sum_hits) / (*h_non_zero_count) : 0.0f;
                std::cerr << "[" << std::setw(3) << total_iterations + 1 << "/" << phi_samples * theta_samples << "]"
                    << " | Θ: " << std::setw(3) << (int)theta_deg << "°"
                    << " | Φ: " << std::setw(3) << (int)phi_deg << "°"
                    << " | " << formatTime(iter_time) 
                    << " | Hit %: " << std::setprecision(1) << 100.0f - (100.0f * (*h_zero_count) / N)  << "%" 
                    << " | Avg of bounces: " << std::fixed << std::setprecision(2) << avg_hits
                    << " | Max bounces: " << *h_max_hits << " (" << max_bounces << ")\n";
            } else {
                std::cerr << "[" << std::setw(3) << total_iterations + 1 << "/" << phi_samples * theta_samples << "]"
                    << " | Θ: " << std::setw(3) << (int)theta_deg << "°"
                    << " | Φ: " << std::setw(3) << (int)phi_deg << "°"
                    << " | " << formatTime(iter_time) << "\n";
            }
            total_iterations++;
        }
    }
    
    clock_t total_sweep_end = clock();
    double total_time = double(total_sweep_end - total_sweep_start) / CLOCKS_PER_SEC;
    printEndSeparator();

    printSeparator("PERFORMANCE SUMMARY");
    printKV(" Total Sweep Time", formatTime(total_time));
    printKV(" Total Points", total_iterations);
    printKV(" Average Time/Point", formatTime(total_time / total_iterations));
    printEndSeparator();

    printSeparator("CLEANUP");
    free_world<<<1,1>>>(d_list, d_world);
    cudaDeviceSynchronize();
    
    cudaFree(d_hit_pos); cudaFree(d_hit_normal); cudaFree(d_hit_dir); cudaFree(d_hit_pol);
    cudaFree(d_hit_dist); cudaFree(d_hit_count); 
    cudaFree(d_accum_theta); cudaFree(d_accum_phi);
    cudaFree(d_list); cudaFree(d_world);
    cudaFree(d_max_hits); cudaFree(d_zero_count); cudaFree(d_sum_hits); cudaFree(d_non_zero_count);
    
    cudaFreeHost(h_accum_theta); cudaFreeHost(h_accum_phi);
    cudaFreeHost(h_max_hits); cudaFreeHost(h_zero_count); cudaFreeHost(h_sum_hits); cudaFreeHost(h_non_zero_count);
    cudaStreamDestroy(compute_stream);
    
    std::cerr << "│  Device and host memory released. Success.\n";
    printEndSeparator();

    return 0;
}