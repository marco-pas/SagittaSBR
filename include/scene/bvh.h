#ifndef SCENE_BVH_H
#define SCENE_BVH_H

#include "RT/vec3.h"

struct alignas(16) Triangle {
    vec3 v0;
    vec3 v1;
    vec3 v2;
};

struct alignas(16) BvhNode {
    vec3 boundsMin;
    vec3 boundsMax;
    int left;
    int right;
    int firstTri;
    int triCount;
};

struct bvhGpuData {
    Triangle* triangles = nullptr;
    BvhNode* nodes = nullptr;
    int triangleCount = 0;
    int nodeCount = 0;
    int rootIndex = 0;
};

#endif
