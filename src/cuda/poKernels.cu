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

    float phase = 2.0f * k * hitDist[idx];
    phase = fmodf(phase, 2.0f * M_PI);

    float mag = (k * rayArea / (4.0f * M_PI))
            * 2.0f
            * cosTheta
            * reflectionConst;

    cuFloatComplex contrib = make_cuFloatComplex(-mag * sinf(phase), mag * cosf(phase));

    atomicAdd(&accum->x, contrib.x);
    atomicAdd(&accum->y, contrib.y);
}