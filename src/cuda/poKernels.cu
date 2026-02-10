#include "cuda/poKernels.cuh"
#include "cuda/gpuConfig.cuh"

#include <cmath>

#if defined(USE_HIP)
#define cuCmulf hipCmulf
#endif

// The biggest performance difference between NVIDIA and AMD comes from PO integration
// Issue comes from different warp size, which slows down reductions.

// Use centralized GPU configuration from gpuConfig.cuh
// - GPU_WARP_SIZE: 64 (AMD) or 32 (NVIDIA)
// - GPU_SHFL_DOWN: Platform-specific shuffle intrinsic
// - GPU_MAX_WARPS_PER_BLOCK: 4 (AMD) or 8 (NVIDIA)

// Warp-level reduction for Real
// Uses GPU_SHFL_DOWN from gpuConfig.cuh for platform-specific shuffle
__device__ __forceinline__ Real warpReduceSum(Real val) {
#if GPU_WARP_SIZE == 64
    // AMD: wavefront size 64 - need extra reduction step
    val += GPU_SHFL_DOWN(val, 32);
#endif
    // Common reduction steps for both platforms
    val += GPU_SHFL_DOWN(val, 16);
    val += GPU_SHFL_DOWN(val, 8);
    val += GPU_SHFL_DOWN(val, 4);
    val += GPU_SHFL_DOWN(val, 2);
    val += GPU_SHFL_DOWN(val, 1);
    return val;
}

// Block-level reduction using shared memory
// Uses GPU_MAX_WARPS_PER_BLOCK and GPU_WARP_SIZE from gpuConfig.cuh
__device__ __forceinline__ Real blockReduceSum(Real val) {
    __shared__ Real shared[GPU_MAX_WARPS_PER_BLOCK];
    
    unsigned int lane = threadIdx.x % GPU_WARP_SIZE;
    unsigned int warpId = threadIdx.x / GPU_WARP_SIZE;
    unsigned int numWarps = (blockDim.x + GPU_WARP_SIZE - 1) / GPU_WARP_SIZE;
    
    // First reduce within warp
    val = warpReduceSum(val);
    
    // First lane of each warp writes to shared memory
    if (lane == 0) {
        shared[warpId] = val;
    }
    
    __syncthreads();
    
    // First warp reduces the partial sums
    // Only read if this warp slot was actually used
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : REAL_CONST(0.0);
    
    if (warpId == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// Physical Optics Integral

__global__ void integratePoMultiBounce(
    const vec3* __restrict__ hitNormal,
    const vec3* __restrict__ lastDir,
    const Real* __restrict__ hitDist,
    const int* __restrict__ hitCount,
    int n, Real k, vec3 kInc, Real rayArea,
    cuRealComplex* __restrict__ accum,
    Real reflectionConst) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes its local contribution
    Real localReal = REAL_CONST(0.0);
    Real localImag = REAL_CONST(0.0);

    if (idx < n && hitCount[idx] > 0) {
        // Integer exponentiation: faster than devPow for small bounce counts (1–20)
        Real totalReflCoeff = REAL_CONST(1.0);
        for (int b = 0; b < hitCount[idx]; ++b) {
            totalReflCoeff *= reflectionConst;
        }

        // Use the last-bounce incoming ray direction for the obliquity factor.
        // For single-bounce rays, lastDir == kInc so this is equivalent.
        // For multi-bounce rays, lastDir is the direction arriving at the
        // exit surface after all prior reflections.
        Real cosTheta = dot(hitNormal[idx], -lastDir[idx]);
        
        if (cosTheta > REAL_CONST(0.0)) {
            Real phase = REAL_CONST(2.0) * k * hitDist[idx];
            phase = devFmod(phase, REAL_CONST(2.0) * Real(M_PI));

            Real mag = (k * rayArea / (REAL_CONST(4.0) * Real(M_PI)))
                    * REAL_CONST(2.0)
                    * cosTheta
                    * totalReflCoeff;

            // Use devSinCos for efficiency
            Real sinPhase, cosPhase;
            devSinCos(phase, &sinPhase, &cosPhase);
            
            localReal = -mag * sinPhase;
            localImag = mag * cosPhase;
        }
    }

    // Block-level reduction using shared memory and warp shuffles
    Real blockSumReal = blockReduceSum(localReal);
    Real blockSumImag = blockReduceSum(localImag);

    // Only thread 0 of each block does atomicAdd to global memory
    // This reduces atomic contention from N threads to N/blockSize blocks
    if (threadIdx.x == 0) {
        atomicAdd(&accum->x, blockSumReal);
        atomicAdd(&accum->y, blockSumImag);
    }
}