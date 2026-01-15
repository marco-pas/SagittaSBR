#include "scene/bvhBuilder.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace {
vec3 minVec3(const vec3& a, const vec3& b) {
    return vec3(std::fmin(a.x(), b.x()),
                std::fmin(a.y(), b.y()),
                std::fmin(a.z(), b.z()));
}

vec3 maxVec3(const vec3& a, const vec3& b) {
    return vec3(std::fmax(a.x(), b.x()),
                std::fmax(a.y(), b.y()),
                std::fmax(a.z(), b.z()));
}

vec3 triangleMin(const Triangle& tri) {
    return minVec3(minVec3(tri.v0, tri.v1), tri.v2);
}

vec3 triangleMax(const Triangle& tri) {
    return maxVec3(maxVec3(tri.v0, tri.v1), tri.v2);
}

vec3 triangleCentroid(const Triangle& tri) {
    return (tri.v0 + tri.v1 + tri.v2) / 3.0f;
}

int maxAxis(const vec3& v) {
    if (v.x() >= v.y() && v.x() >= v.z()) {
        return 0;
    }
    if (v.y() >= v.z()) {
        return 1;
    }
    return 2;
}

int buildNode(std::vector<Triangle>& triangles,
              std::vector<BvhNode>& nodes,
              int start,
              int end,
              int leafSize) {
    vec3 boundsMin(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 boundsMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = start; i < end; ++i) {
        boundsMin = minVec3(boundsMin, triangleMin(triangles[i]));
        boundsMax = maxVec3(boundsMax, triangleMax(triangles[i]));
    }

    BvhNode node{};
    node.boundsMin = boundsMin;
    node.boundsMax = boundsMax;
    node.left = -1;
    node.right = -1;
    node.firstTri = start;
    node.triCount = end - start;

    int nodeIndex = static_cast<int>(nodes.size());
    nodes.push_back(node);

    if (node.triCount <= leafSize) {
        return nodeIndex;
    }

    vec3 extent = boundsMax - boundsMin;
    int axis = maxAxis(extent);
    int mid = start + (node.triCount / 2);

    std::nth_element(
        triangles.begin() + start,
        triangles.begin() + mid,
        triangles.begin() + end,
        [axis](const Triangle& a, const Triangle& b) {
            vec3 ca = triangleCentroid(a);
            vec3 cb = triangleCentroid(b);
            return ca[axis] < cb[axis];
        });

    int left = buildNode(triangles, nodes, start, mid, leafSize);
    int right = buildNode(triangles, nodes, mid, end, leafSize);

    nodes[nodeIndex].left = left;
    nodes[nodeIndex].right = right;
    nodes[nodeIndex].firstTri = 0;
    nodes[nodeIndex].triCount = 0;

    return nodeIndex;
}
}

bvhBuildResult buildBvh(std::vector<Triangle> triangles, int leafSize) {
    bvhBuildResult result;
    result.triangles = std::move(triangles);
    result.nodes.clear();

    if (!result.triangles.empty()) {
        result.rootIndex = buildNode(
            result.triangles,
            result.nodes,
            0,
            static_cast<int>(result.triangles.size()),
            leafSize);
    }

    return result;
}
