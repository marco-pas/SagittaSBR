#include "app/deviceBuffers.hpp"

#include "cuda/cudaUtils.cuh"

void allocateDeviceBuffers(deviceBuffers& buffers, int count) {
    buffers.count = count;
    checkCudaErrors(cudaMallocManaged(&buffers.hitPos, count * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&buffers.hitNormal, count * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&buffers.hitDist, count * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&buffers.hitFlag, count * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&buffers.hitCount, count * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&buffers.accum, sizeof(cuFloatComplex)));
}

void freeDeviceBuffers(deviceBuffers& buffers) {
    checkCudaErrors(cudaFree(buffers.hitPos));
    checkCudaErrors(cudaFree(buffers.hitNormal));
    checkCudaErrors(cudaFree(buffers.hitDist));
    checkCudaErrors(cudaFree(buffers.hitFlag));
    checkCudaErrors(cudaFree(buffers.hitCount));
    checkCudaErrors(cudaFree(buffers.accum));

    buffers = deviceBuffers{};
}
