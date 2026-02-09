#include "cuda/rtKernels.cuh"

#include <cfloat>
#include <cmath>

#include "RT/hitable.h"
#include "RT/ray.h"

namespace {
__device__ bool hitTriangle(const Triangle& tri, const ray& r, Real tMin, Real tMax,
                            hitRecord& rec) {
    const Real eps = REAL_CONST(1.0e-6);
    vec3 edge1 = tri.v1 - tri.v0;
    vec3 edge2 = tri.v2 - tri.v0;
    vec3 pvec = cross(r.direction(), edge2);
    Real det = dot(edge1, pvec);

    if (devAbs(det) < eps) {
        return false;
    }

    Real invDet = REAL_CONST(1.0) / det;
    vec3 tvec = r.origin() - tri.v0;
    Real u = dot(tvec, pvec) * invDet;
    if (u < REAL_CONST(0.0) || u > REAL_CONST(1.0)) {
        return false;
    }

    vec3 qvec = cross(tvec, edge1);
    Real v = dot(r.direction(), qvec) * invDet;
    if (v < REAL_CONST(0.0) || u + v > REAL_CONST(1.0)) {
        return false;
    }

    Real t = dot(edge2, qvec) * invDet;
    if (t < tMin || t > tMax) {
        return false;
    }

    rec.t = t;
    rec.p = r.pointAtParameter(t);
    rec.normal = unitVector(cross(edge1, edge2));
    return true;
}

__device__ bool hitAabb(const BvhNode& node, const ray& r, Real tMin, Real tMax) {
    vec3 invDir(REAL_CONST(1.0) / r.direction().x(),
                REAL_CONST(1.0) / r.direction().y(),
                REAL_CONST(1.0) / r.direction().z());

    vec3 t0 = (node.boundsMin - r.origin()) * invDir;
    vec3 t1 = (node.boundsMax - r.origin()) * invDir;

    Real tmin = devFmax(devFmax(devFmin(t0.x(), t1.x()), devFmin(t0.y(), t1.y())), devFmin(t0.z(), t1.z()));
    Real tmax = devFmin(devFmin(devFmax(t0.x(), t1.x()), devFmax(t0.y(), t1.y())), devFmax(t0.z(), t1.z()));

    tmin = devFmax(tmin, tMin);
    tmax = devFmin(tmax, tMax);

    return tmax >= tmin;
}

__device__ bool hitBvh(const Triangle* triangles, const BvhNode* nodes, int rootIndex,
                       const ray& r, Real tMin, Real tMax, hitRecord& rec) {
    constexpr int stackSize = 64;
    int stack[stackSize];
    int stackPtr = 0;
    stack[stackPtr++] = rootIndex;

    bool hitAnything = false;
    Real closest = tMax;
    hitRecord tempRec;

    while (stackPtr > 0) {
        int nodeIndex = stack[--stackPtr];
        const BvhNode& node = nodes[nodeIndex];

        if (!hitAabb(node, r, tMin, closest)) {
            continue;
        }

        if (node.triCount > 0) {
            for (int i = 0; i < node.triCount; ++i) {
                int triIndex = node.firstTri + i;
                if (hitTriangle(triangles[triIndex], r, tMin, closest, tempRec)) {
                    hitAnything = true;
                    closest = tempRec.t;
                    rec = tempRec;
                }
            }
        } else {
            if (stackPtr + 2 <= stackSize) {
                stack[stackPtr++] = node.left;
                stack[stackPtr++] = node.right;
            }
        }
    }

    return hitAnything;
}
}

// __global__ void launchRaysMultiBounce(vec3* hitPos, vec3* hitNormal,
//                                       float* hitDist, int* hitCount, int nx,
//                                       int ny, vec3 llc, vec3 horiz, vec3 vert,
//                                       vec3 rayDir, const Triangle* triangles,
//                                       const BvhNode* nodes, int rootIndex,
//                                       int maxBounces) {

//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i >= nx || j >= ny) {
//         return;
//     }
//     int idx = j * nx + i;

//     float u = (i + 0.5f) / float(nx);
//     float v = (j + 0.5f) / float(ny);

//     vec3 currentOrigin = llc + u * horiz + v * vert;
//     vec3 currentDir = rayDir;

//     hitDist[idx] = 0.0f;
//     hitCount[idx] = 0;

//     const float EPS = 1e-4f;

//     for (int k = 0; k < maxBounces; k++) {
//         ray r(currentOrigin, currentDir);
//         hitRecord rec;

//         if (hitBvh(triangles, nodes, rootIndex, r, 0.001f, FLT_MAX, rec)) {
//         if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {

//             hitCount[idx]++;
//             hitPos[idx] = rec.p;
//             hitNormal[idx] = rec.normal;
//             hitDist[idx] += rec.t;

//             current_dir = unit_vector(
//                 currentDir - 2.0f * dot(currentDir, rec.normal) * rec.normal
//             );

//             currentOrigin = rec.p + rec.normal * EPS;

//             // Optional grazing-angle escape
//             // if (fabsf(dot(current_dir, rec.normal)) < 1e-5f)
//             //    break;
//             // }

//             currentOrigin = rec.p + rec.normal * 0.001f;
//             currentDir = currentDir - 2.0f * dot(currentDir, rec.normal) * rec.normal;
//         } else {
//             break;
//         }
//     }
// }



__global__ void launchRaysMultiBounce(vec3* hitPos, vec3* hitNormal,
                                      vec3* lastDir,
                                      Real* hitDist, int* hitCount, int nx,
                                      int ny, vec3 llc, vec3 horiz, vec3 vert,
                                      vec3 rayDir, const Triangle* triangles,
                                      const BvhNode* nodes, int rootIndex,
                                      int maxBounces) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;
    
    int idx = j * nx + i;

    Real u = (i + REAL_CONST(0.5)) / Real(nx);
    Real v = (j + REAL_CONST(0.5)) / Real(ny);

    vec3 currentOrigin = llc + u * horiz + v * vert;
    vec3 currentDir = rayDir;

    hitDist[idx] = REAL_CONST(0.0);
    hitCount[idx] = 0;

    const Real EPS = REAL_CONST(1e-4);

    for (int k = 0; k < maxBounces; k++) {
        ray r(currentOrigin, currentDir);
        hitRecord rec;

        // Use the BVH traversal function provided in your arguments
        if (hitBvh(triangles, nodes, rootIndex, r, REAL_CONST(0.001), REAL_CONST(1e20), rec)) {
            
            // Store hit data on every bounce so that after the loop
            // these arrays hold the last-bounce values needed for PO
            hitPos[idx] = rec.p;
            hitNormal[idx] = rec.normal;
            lastDir[idx] = currentDir;
            
            hitCount[idx]++;
            hitDist[idx] += rec.t;

            // Reflect the ray: R = I - 2 * dot(I, N) * N
            currentDir = unitVector(
                currentDir - REAL_CONST(2.0) * dot(currentDir, rec.normal) * rec.normal
            );

            // Offset the new origin along the normal to prevent self-intersection
            currentOrigin = rec.p + rec.normal * EPS;
            
        } else {
            // Ray escaped into the void
            break;
        }
    }
}