#include "cuda/rtKernels.cuh"

#include <cfloat>

#include "RT/ray.h"

__global__ void launchRays(vec3* hitPos, vec3* hitNormal, float* hitDist,
                           int* hitFlag, int nx, int ny, vec3 llc, vec3 horiz,
                           vec3 vert, vec3 rayDir, hitable** world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }
    int idx = j * nx + i;

    float u = (i + 0.5f) / float(nx);
    float v = (j + 0.5f) / float(ny);

    vec3 rayStart = llc + u * horiz + v * vert;
    ray r(rayStart, rayDir);

    hitRecord rec;
    if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
        hitFlag[idx] = 1;
        hitPos[idx] = rec.p;
        hitNormal[idx] = rec.normal;
        hitDist[idx] = rec.t;
    } else {
        hitFlag[idx] = 0;
    }
}

__global__ void launchRaysMultiBounce(vec3* hitPos, vec3* hitNormal,
                                      float* hitDist, int* hitCount, int nx,
                                      int ny, vec3 llc, vec3 horiz, vec3 vert,
                                      vec3 rayDir, hitable** world,
                                      int maxBounces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }
    int idx = j * nx + i;

    float u = (i + 0.5f) / float(nx);
    float v = (j + 0.5f) / float(ny);

    vec3 currentOrigin = llc + u * horiz + v * vert;
    vec3 currentDir = rayDir;

    hitDist[idx] = 0.0f;
    hitCount[idx] = 0;

    for (int k = 0; k < maxBounces; k++) {
        ray r(currentOrigin, currentDir);
        hitRecord rec;

        if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
            hitCount[idx]++;

            if (k == 0) {
                hitPos[idx] = rec.p;
                hitNormal[idx] = rec.normal;
            }

            hitDist[idx] += rec.t;

            currentOrigin = rec.p;
            currentDir = currentDir - 2.0f * dot(currentDir, rec.normal) * rec.normal;
        } else {
            break;
        }
    }
}
