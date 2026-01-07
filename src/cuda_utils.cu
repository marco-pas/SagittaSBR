/*
 * Implementation of CUDA utility functions.
 */

#include "cuda_utils.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

void check_cuda(cudaError_t result, char const* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line
                  << " '" << func << "'\n";
        cudaDeviceReset();
        exit(1);
    }
}

void printGPUInfo() {
    cudaDeviceProp prop;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found\n";
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
