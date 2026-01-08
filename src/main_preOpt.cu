/*
 * Main entry point for monostatic RCS sweep simulation using CUDA ray tracing.
 * Implements the Shooting and Bouncing Rays (SBR) method with Physical Optics integration.
 */

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
    // mkdir failed — check why
    return errno == EEXIST;
}

int main(int argc, char** argv) {
    std::cerr << "╔════════════════════════════════════════════════════╗\n";
    std::cerr << "║  SagittaSBR  >>---->  MONOSTATIC RCS CALCULATION   ║\n";
    std::cerr << "╚════════════════════════════════════════════════════╝\n";
    
    // GPU info
    printGPUInfo(); 

    // Load configuration
    auto cfg = loadConfig("config.txt");

    // Parse configuration parameters (use defaults if not found)
    float phi_start = cfg.count("phi_start") ? cfg["phi_start"] : 0.0f;
    float phi_end   = cfg.count("phi_end")   ? cfg["phi_end"]   : 180.0f;
    int phi_samples = cfg.count("phi_samples") ? (int)cfg["phi_samples"] : 181;
    float theta_start = cfg.count("theta_start") ? cfg["theta_start"] : 90.0f;
    float theta_end   = cfg.count("theta_end")   ? cfg["theta_end"]   : 90.0f;
    int theta_samples = cfg.count("theta_samples") ? (int)cfg["theta_samples"] : 1;
    float grid_size = cfg.count("grid_size") ? cfg["grid_size"] : 3.0f;
    int nx          = cfg.count("nx")        ? (int)cfg["nx"]   : 400;
    int ny          = cfg.count("ny")        ? (int)cfg["ny"]   : 400;
    int tpbx        = cfg.count("tpbx")        ? (int)cfg["tpbx"]   : 16;
    int tpby        = cfg.count("tpby")        ? (int)cfg["tpby"]   : 16;
    int max_bounces = cfg.count("max_bounces")    ? (int)cfg["max_bounces"]   : 20;
    float reflection_const = cfg.count("reflection_const")    ? cfg["reflection_const"] : 1.0f;
    
    // Frequency handling: command line > config file > default
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

    // Print simulation parameters
    printSeparator("SIMULATION PARAMETERS");
    printKV("Frequency", freq / 1e9, "GHz");
    printKV("Wavelength", lambda * 1000, "mm");
    printKV("Wave Number k", k, "rad/m");
    printKV("Illumination Area", grid_size * grid_size, "m² (make sure this is larger than your world objects)");
    printKV("Grid Size", std::to_string(nx) + "x" + std::to_string(ny));
    printKV("Total Rays", N);
    printKV("Ray Density", N / (grid_size * grid_size), "rays/m² (make sure this is dense enough)");
    printKV("Ray Area", ray_area, "m²");
    printKV("Phi Range", std::to_string(phi_start) + " to " + std::to_string(phi_end));
    printEndSeparator();

    // Memory allocation on device
    printSeparator("MEMORY ALLOCATION");
    vec3 *hit_pos, *hit_normal; 
    float *hit_dist; 
    int *hit_flag, *hit_count; 
    cuFloatComplex *accum;
    checkCudaErrors(cudaMallocManaged(&hit_pos, N * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&hit_normal, N * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&hit_dist, N * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&hit_flag, N * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&hit_count, N * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&accum, sizeof(cuFloatComplex)));

    // Create world of hitable objects
    hitable **d_list, **d_world;
    checkCudaErrors(cudaMalloc(&d_list, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hitable*)));
    create_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "│  Managed memory and world setup complete.\n";
    printEndSeparator();

    // Create output directory
    std::string outputDir = "output";
    if (!createDirectory(outputDir)) {
        std::cerr << "Warning: Could not create output directory '" << outputDir 
                  << "'. Saving results to current directory.\n";
        outputDir = ".";
    } else {
        std::cerr << "Created output directory: " << outputDir << "\n";
    }

    // Output file for results
    std::string outputFile = outputDir + "/rcs_results.csv";
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open output file '" << outputFile << "'\n";
        return 1;
    }
    
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

    // Define threads and blocks for kernel launches

    dim3 threads(tpbx, tpby);
    dim3 blocks((nx + tpbx - 1) / tpbx, (ny + tpby - 1) / tpby);
    

    // Execute monostatic RCS sweep
    printSeparator("EXECUTING SWEEP");
    clock_t total_sweep_start = clock();
    int total_iterations = 0;

    // Theta loop
    for (int t = 0; t < theta_samples; ++t) {
        float theta_deg = theta_start + (theta_samples > 1 ? t * (theta_end - theta_start) / (theta_samples - 1) : 0);
        float theta_rad = theta_deg * M_PI / 180.0f;

        // Phi loop
        for (int p = 0; p < phi_samples; ++p) {
            clock_t iter_start = clock();
            
            float phi_deg = phi_start + (phi_samples > 1 ? p * (phi_end - phi_start) / (phi_samples - 1) : 0);
            float phi_rad = phi_deg * M_PI / 180.0f;

            // Setup ray grid geometry
            vec3 dir(sinf(theta_rad)*cosf(phi_rad), sinf(theta_rad)*sinf(phi_rad), cosf(theta_rad));
            vec3 ray_dir = -dir;

            vec3 up(0, 1, 0);
            if (fabs(dir.y()) > 0.9f) up = vec3(1, 0, 0);
            vec3 u_vec = unit_vector(cross(up, dir)) * grid_size;
            vec3 v_vec = unit_vector(cross(dir, u_vec)) * grid_size;
            vec3 llc = (dir * distance_offset) - 0.5f * u_vec - 0.5f * v_vec;

            accum->x = 0.0f; 
            accum->y = 0.0f;

            // @@ Launch rays (choose single or multi-bounce)

            // Single reflection:
            // launcher<<<blocks, threads>>>(hit_pos, hit_normal, hit_dist, hit_flag, nx, ny, llc, u_vec, v_vec, ray_dir, d_world);
            
            // Multiple reflections:
            launcher_multibounce<<<blocks, threads>>>(hit_pos, hit_normal, hit_dist, hit_count, nx, ny, llc, u_vec, v_vec, ray_dir, d_world, max_bounces);
            cudaDeviceSynchronize();

            // @@ Compute PO integral for scattered field

            // Single reflection:
            // integral_PO<<<(N+255)/256, 256>>>(hit_pos, hit_normal, hit_dist, hit_flag, N, k, ray_dir, ray_area, accum);
            
            // Multiple reflections:
            integral_PO_multibounce<<<(N+255)/256, 256>>>(hit_pos, hit_normal, hit_dist, hit_count, N, k, ray_dir, ray_area, accum, reflection_const);
            cudaDeviceSynchronize();

            // Compute RCS
            float sigma = 4.0f * M_PI * (accum->x * accum->x + accum->y * accum->y);    // RCS in m^2
            float rcs_dBsm = 10.0f * log10f(fmaxf(sigma, 1e-10f));                      // RCS in dBsm

            // Write to CSV
            outFile << theta_deg << "," << phi_deg << "," << sigma << "," << rcs_dBsm << "\n";
            
            clock_t iter_end = clock();
            double iter_time = double(iter_end - iter_start) / CLOCKS_PER_SEC;

            // Compute hit statistics
            int max_h = 0;
            int zero_count = 0;
            long long sum_hits = 0;
            int non_zero_count = 0;

            for (int n = 0; n < N; n++) {
                int val = hit_count[n];
                if (val == 0) {
                    zero_count++;
                } else {
                    if (val > max_h) max_h = val;
                    sum_hits += val;
                    non_zero_count++;
                }
            }

            float avg_hits = (non_zero_count > 0) ? (float)sum_hits / non_zero_count : 0.0f;

            // Print progress
            std::cerr << "[" << std::setw(3) << total_iterations + 1 << "/" << phi_samples * theta_samples << "]"
                << " | Θ: " << std::setw(3) << (int)theta_deg << "°"
                << " | Φ: " << std::setw(3) << (int)phi_deg << "°"
                << " | " << formatTime(iter_time) 
                << " | Hit %: " << std::setprecision(1) << 100.0f - (100.0f * zero_count / N)  << "%" 
                << " | Avg of bounces: " << std::fixed << std::setprecision(2) << avg_hits
                << " | Max bounces: " << max_h << " (" << max_bounces << ")"
                << "\n";
                        
            total_iterations++;
        }
    }
    
    clock_t total_sweep_end = clock();
    double total_time = double(total_sweep_end - total_sweep_start) / CLOCKS_PER_SEC;
    printEndSeparator();

    // Performance summary
    printSeparator("PERFORMANCE SUMMARY");
    printKV(" Total Sweep Time", formatTime(total_time));
    printKV(" Total Points", total_iterations);
    printKV(" Average Time/Point", formatTime(total_time / total_iterations));
    printEndSeparator();

    // Cleanup
    printSeparator("CLEANUP");
    free_world<<<1,1>>>(d_list, d_world);
    cudaDeviceSynchronize();
    cudaFree(hit_pos); cudaFree(hit_normal); cudaFree(hit_dist);
    cudaFree(hit_flag); cudaFree(accum); cudaFree(d_list); cudaFree(d_world);
    cudaFree(hit_count);
    std::cerr << "│  Device memory released. Success.\n";
    printEndSeparator();

    return 0;
}
