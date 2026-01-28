#include "cuda/poKernels.cuh"
#include "cuda/gpuConfig.cuh"

#include <cmath>

#if defined(USE_HIP)
#define cuCmulf hipCmulf
#endif

// Use centralized GPU configuration from gpuConfig.cuh
// - GPU_WARP_SIZE: 64 (AMD) or 32 (NVIDIA)
// - GPU_SHFL_DOWN: Platform-specific shuffle intrinsic
// - GPU_MAX_WARPS_PER_BLOCK: 4 (AMD) or 8 (NVIDIA)

// Warp-level reduction for float
// Uses GPU_SHFL_DOWN from gpuConfig.cuh for platform-specific shuffle
__device__ __forceinline__ float warpReduceSum(float val) {
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
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[GPU_MAX_WARPS_PER_BLOCK];
    
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
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    
    if (warpId == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

__global__ void integratePo(vec3* /* hitPos */, vec3* hitNormal, float* hitDist,
                            int* hitFlag, int n, float k, vec3 kInc,
                            float rayArea, cuFloatComplex* accum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float localReal = 0.0f;
    float localImag = 0.0f;

    if (idx < n && hitFlag[idx]) {
        float cosTheta = dot(hitNormal[idx], -kInc);
        if (cosTheta > 0.0f) {
            float phase = 2.0f * k * hitDist[idx];
            float mag = (k * rayArea / (4.0f * M_PI)) * 2.0f;
            float sinPhase, cosPhase;
            sincosf(phase, &sinPhase, &cosPhase);
            localReal = -mag * sinPhase;
            localImag = mag * cosPhase;
        }
    }

    // Block-level reduction
    float blockSumReal = blockReduceSum(localReal);
    float blockSumImag = blockReduceSum(localImag);

    // Only thread 0 does the atomicAdd (one per block instead of one per thread)
    if (threadIdx.x == 0) {
        atomicAdd(&accum->x, blockSumReal);
        atomicAdd(&accum->y, blockSumImag);
    }
}

__global__ void integratePoMultiBounce(
    const vec3* __restrict__ /* hitPos */,
    const vec3* __restrict__ hitNormal,
    const float* __restrict__ hitDist,
    const int* __restrict__ hitCount,
    int n, float k, vec3 kInc, float rayArea,
    cuFloatComplex* __restrict__ accum,
    float reflectionConst) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes its local contribution
    float localReal = 0.0f;
    float localImag = 0.0f;

    if (idx < n && hitCount[idx] > 0) {
        float totalReflCoeff = powf(reflectionConst, static_cast<float>(hitCount[idx]));
        float cosTheta = dot(hitNormal[idx], -kInc);
        
        if (cosTheta > 0.0f) {
            float phase = 2.0f * k * hitDist[idx];
            phase = fmodf(phase, 2.0f * M_PI);

            float mag = (k * rayArea / (4.0f * M_PI))
                    * 2.0f
                    * cosTheta
                    * totalReflCoeff;

            // Use sincosf for efficiency
            float sinPhase, cosPhase;
            sincosf(phase, &sinPhase, &cosPhase);
            
            localReal = -mag * sinPhase;
            localImag = mag * cosPhase;
        }
    }

    // Block-level reduction using shared memory and warp shuffles
    float blockSumReal = blockReduceSum(localReal);
    float blockSumImag = blockReduceSum(localImag);

    // Only thread 0 of each block does atomicAdd to global memory
    // This reduces atomic contention from N threads to N/blockSize blocks
    if (threadIdx.x == 0) {
        atomicAdd(&accum->x, blockSumReal);
        atomicAdd(&accum->y, blockSumImag);
    }
}

// __global__ void integratePoMultiBounce(vec3* hitPos, vec3* hitNormal,
//                                        float* hitDist, int* hitCount, int n,
//                                        float k, vec3 kInc, float rayArea,
//                                        cuFloatComplex* accum,
//                                        float reflectionConst) {

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx >= n || hitCount[idx] == 0) {
//         return;
//     }

//     // All calculations in double precision
//     double totalReflCoeff = pow(static_cast<double>(reflectionConst), 
//                                 static_cast<double>(hitCount[idx]));

//     double cosTheta = dot(hitNormal[idx], -kInc);  // Assuming vec3 dot returns double or cast it
//     if (cosTheta <= 0.0) {
//         return;
//     }

//     // Phase calculation in double precision
//     double k_d = static_cast<double>(k);
//     double hitDist_d = static_cast<double>(hitDist[idx]);
//     double phase = 2.0 * k_d * hitDist_d;
//     phase = fmod(phase, 2.0 * M_PI);
    
//     // Magnitude calculation in double precision
//     double rayArea_d = static_cast<double>(rayArea);
//     double mag = (k_d * rayArea_d / (4.0 * M_PI))
//             * 2.0
//             * cosTheta
//             * totalReflCoeff;

//     // Trig functions in double precision
//     double sinVal, cosVal;
//     sincos(phase, &sinVal, &cosVal);  // Double precision sincos
    
//     // Convert to single precision for output
//     cuFloatComplex contrib = make_cuFloatComplex(
//         static_cast<float>(-mag * sinVal), 
//         static_cast<float>(mag * cosVal)
//     );

//     atomicAdd(&accum->x, contrib.x);
//     atomicAdd(&accum->y, contrib.y);
// }