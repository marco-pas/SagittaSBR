#include "app/deviceBuffers.hpp"

#include "cuda/cudaUtils.cuh"

// AMD GPUs have very poor performance with cudaMallocManaged (unified memory).
// Use explicit device memory allocation for GPU-resident data and separate
// host-pinned memory for data that needs CPU access.
// 
// The hitCount array needs CPU access for statistics, so we use pinned memory
// and explicit copies. The accum buffer uses device memory with explicit copies.

void allocateDeviceBuffers(deviceBuffers& buffers, int count) {
    buffers.count = count;
    
    // GPU-only buffers - use device memory (fast on both NVIDIA and AMD)
    checkCudaErrors(cudaMalloc(&buffers.hitNormal, count * sizeof(vec3)));
    checkCudaErrors(cudaMalloc(&buffers.lastDir, count * sizeof(vec3)));
    checkCudaErrors(cudaMalloc(&buffers.hitDist, count * sizeof(Real)));
    checkCudaErrors(cudaMalloc(&buffers.hitCount, count * sizeof(int)));
    
    // Accumulator - device memory for atomic operations
    checkCudaErrors(cudaMalloc(&buffers.accum, sizeof(cuRealComplex)));
    
    // Allocate host buffer for hitCount stats (only used when showHitStats is true)
    checkCudaErrors(cudaMallocHost(&buffers.hitCountHost, count * sizeof(int)));
    
    // Host-side accumulator for reading results
    checkCudaErrors(cudaMallocHost(&buffers.accumHost, sizeof(cuRealComplex)));
}

void freeDeviceBuffers(deviceBuffers& buffers) {
    checkCudaErrors(cudaFree(buffers.hitNormal));
    checkCudaErrors(cudaFree(buffers.lastDir));
    checkCudaErrors(cudaFree(buffers.hitDist));
    checkCudaErrors(cudaFree(buffers.hitCount));
    checkCudaErrors(cudaFree(buffers.accum));
    
    checkCudaErrors(cudaFreeHost(buffers.hitCountHost));
    checkCudaErrors(cudaFreeHost(buffers.accumHost));

    buffers = deviceBuffers{};
}
