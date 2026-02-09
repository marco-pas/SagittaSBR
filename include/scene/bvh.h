#ifndef SCENE_BVH_H
#define SCENE_BVH_H

#include "RT/vec3.h"

template<typename T>
struct alignas(16) Triangle_t {
    vec3_t<T> v0;
    vec3_t<T> v1;
    vec3_t<T> v2;
};

template<typename T>
struct alignas(16) BvhNode_t {
    vec3_t<T> boundsMin;
    vec3_t<T> boundsMax;
    int left;
    int right;
    int firstTri;
    int triCount;
};

// Type aliases
using Triangle = Triangle_t<Real>;
using BvhNode = BvhNode_t<Real>;

struct bvhGpuData {
    Triangle* triangles = nullptr;
    BvhNode* nodes = nullptr;
    int triangleCount = 0;
    int nodeCount = 0;
    int rootIndex = 0;
};

#endif
