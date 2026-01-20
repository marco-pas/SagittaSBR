#include "cuda/poKernels.cuh"

#include <cmath>

__global__ void integratePo(vec3* hitPos, vec3* hitNormal, float* hitDist,
                            int* hitFlag, int n, float k, vec3 kInc,
                            float rayArea, cuFloatComplex* accum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n || !hitFlag[idx]) {
        return;
    }

    float cosTheta = dot(hitNormal[idx], -kInc);
    if (cosTheta <= 0.0f) {
        return;
    }

    float phase = 2.0f * k * hitDist[idx];
    float mag = (k * rayArea / (4.0f * M_PI)) * 2.0f;
    cuFloatComplex contrib = make_cuFloatComplex(-mag * sinf(phase), mag * cosf(phase));

    atomicAdd(&accum->x, contrib.x);
    atomicAdd(&accum->y, contrib.y);
}

__global__ void integratePoMultiBounce(vec3* hitPos, vec3* hitNormal,
                                       float* hitDist, int* hitCount, int n,
                                       float k, vec3 kInc, float rayArea,
                                       cuFloatComplex* accum,
                                       float reflectionConst) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n || hitCount[idx] == 0) {
        return;
    }

    float totalReflCoeff = powf(reflectionConst, static_cast<float>(hitCount[idx]));

    float cosTheta = dot(hitNormal[idx], -kInc);
    if (cosTheta <= 0.0f) {
        return;
    }

    // set up log-scale for k

    float phase = 2.0f * k * hitDist[idx];
    phase = fmodf(phase, 2.0f * M_PI);

    float mag = (k * rayArea / (4.0f * M_PI))
            * 2.0f
            * cosTheta
            * totalReflCoeff;

    cuFloatComplex contrib = make_cuFloatComplex(-mag * sinf(phase), mag * cosf(phase));

    atomicAdd(&accum->x, contrib.x);
    atomicAdd(&accum->y, contrib.y);
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