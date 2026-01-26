#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#include "hip/hipifly.hpp"
#else
#include <cuda_runtime.h>
#endif

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

void checkCuda(cudaError_t result, const char* func, const char* file, int line);
void printGpuInfo();

#endif
