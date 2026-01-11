#include "cuda/cudaUtils.cuh"

#include <cstdlib>
#include <iostream>

void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line
                  << " '" << func << "'\n";
        cudaDeviceReset();
        std::exit(1);
    }
}

void printGpuInfo() {
    cudaDeviceProp prop;
    int deviceCount = 0;
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
    std::cerr << "╚═══════════════════════════════════════════════════╝\n\n";
}
