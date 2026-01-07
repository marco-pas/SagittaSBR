/*
 * CUDA utility functions for error checking and device management.
 * Provides macros and functions for safe CUDA API calls and device queries.
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>

// CUDA error checking macro
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

// Function to check CUDA errors
void check_cuda(cudaError_t result, char const* func, const char* file, int line);

// Print GPU device information
void printGPUInfo();

#endif // CUDA_UTILS_H
