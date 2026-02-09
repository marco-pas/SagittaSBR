#include "scene/bvhBuilder.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
struct buildContext {
    std::vector<Triangle>* triangles;
    std::vector<BvhNode>* nodes;
    std::atomic<int>* nodeCounter;
    bvhBuildOptions options;
};

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
    return (tri.v0 + tri.v1 + tri.v2) / Real(3.0);
}

Real surfaceArea(const vec3& minV, const vec3& maxV) {
    vec3 d = maxV - minV;
    Real dx = std::fmax(Real(0.0), d.x());
    Real dy = std::fmax(Real(0.0), d.y());
    Real dz = std::fmax(Real(0.0), d.z());
    return Real(2.0) * (dx * dy + dy * dz + dz * dx);
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

void computeBounds(const std::vector<Triangle>& triangles, int start, int end,
                   vec3& outMin, vec3& outMax) {
    vec3 boundsMin(Real(REAL_FLT_MAX), Real(REAL_FLT_MAX), Real(REAL_FLT_MAX));
    vec3 boundsMax(Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX));

    for (int i = start; i < end; ++i) {
        boundsMin = minVec3(boundsMin, triangleMin(triangles[i]));
        boundsMax = maxVec3(boundsMax, triangleMax(triangles[i]));
    }

    outMin = boundsMin;
    outMax = boundsMax;
}

void computeCentroidBounds(const std::vector<Triangle>& triangles, int start, int end,
                           vec3& outMin, vec3& outMax) {
    vec3 boundsMin(Real(REAL_FLT_MAX), Real(REAL_FLT_MAX), Real(REAL_FLT_MAX));
    vec3 boundsMax(Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX));

    for (int i = start; i < end; ++i) {
        vec3 c = triangleCentroid(triangles[i]);
        boundsMin = minVec3(boundsMin, c);
        boundsMax = maxVec3(boundsMax, c);
    }

    outMin = boundsMin;
    outMax = boundsMax;
}

bool partitionSimple(std::vector<Triangle>& triangles, int start, int end, int& mid) {
    vec3 centroidMin;
    vec3 centroidMax;
    computeCentroidBounds(triangles, start, end, centroidMin, centroidMax);

    vec3 extent = centroidMax - centroidMin;
    int axis = maxAxis(extent);
    mid = start + ((end - start) / 2);

    std::nth_element(
        triangles.begin() + start,
        triangles.begin() + mid,
        triangles.begin() + end,
        [axis](const Triangle& a, const Triangle& b) {
            vec3 ca = triangleCentroid(a);
            vec3 cb = triangleCentroid(b);
            return ca[axis] < cb[axis];
        });

    return mid > start && mid < end;
}

bool partitionSah(std::vector<Triangle>& triangles, int start, int end, int binCount,
                  const vec3& boundsMin, const vec3& boundsMax, int& mid) {
    struct Bin {
        vec3 boundsMin = vec3(Real(REAL_FLT_MAX), Real(REAL_FLT_MAX), Real(REAL_FLT_MAX));
        vec3 boundsMax = vec3(Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX));
        int count = 0;
    };

    int triCount = end - start;
    vec3 centroidMin;
    vec3 centroidMax;
    computeCentroidBounds(triangles, start, end, centroidMin, centroidMax);

    vec3 centroidExtent = centroidMax - centroidMin;
    Real parentArea = surfaceArea(boundsMin, boundsMax);
    if (parentArea <= Real(0.0)) {
        return false;
    }

    int bestAxis = -1;
    int bestSplit = -1;
    Real bestCost = Real(REAL_FLT_MAX);

    for (int axis = 0; axis < 3; ++axis) {
        Real extent = centroidExtent[axis];
        if (extent <= Real(1.0e-6)) {
            continue;
        }

        std::vector<Bin> bins(static_cast<std::size_t>(binCount));

        for (int i = start; i < end; ++i) {
            vec3 c = triangleCentroid(triangles[i]);
            Real offset = (c[axis] - centroidMin[axis]) / extent;
            int binIndex = static_cast<int>(offset * binCount);
            if (binIndex < 0) {
                binIndex = 0;
            } else if (binIndex >= binCount) {
                binIndex = binCount - 1;
            }

            Bin& bin = bins[binIndex];
            bin.boundsMin = minVec3(bin.boundsMin, triangleMin(triangles[i]));
            bin.boundsMax = maxVec3(bin.boundsMax, triangleMax(triangles[i]));
            bin.count += 1;
        }

        std::vector<vec3> leftMin(static_cast<std::size_t>(binCount));
        std::vector<vec3> leftMax(static_cast<std::size_t>(binCount));
        std::vector<int> leftCount(static_cast<std::size_t>(binCount));

        std::vector<vec3> rightMin(static_cast<std::size_t>(binCount));
        std::vector<vec3> rightMax(static_cast<std::size_t>(binCount));
        std::vector<int> rightCount(static_cast<std::size_t>(binCount));

        vec3 runningMin(Real(REAL_FLT_MAX), Real(REAL_FLT_MAX), Real(REAL_FLT_MAX));
        vec3 runningMax(Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX));
        int runningCount = 0;
        for (int i = 0; i < binCount; ++i) {
            if (bins[i].count > 0) {
                runningMin = minVec3(runningMin, bins[i].boundsMin);
                runningMax = maxVec3(runningMax, bins[i].boundsMax);
                runningCount += bins[i].count;
            }
            leftMin[i] = runningMin;
            leftMax[i] = runningMax;
            leftCount[i] = runningCount;
        }

        runningMin = vec3(Real(REAL_FLT_MAX), Real(REAL_FLT_MAX), Real(REAL_FLT_MAX));
        runningMax = vec3(Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX), Real(-REAL_FLT_MAX));
        runningCount = 0;
        for (int i = binCount - 1; i >= 0; --i) {
            if (bins[i].count > 0) {
                runningMin = minVec3(runningMin, bins[i].boundsMin);
                runningMax = maxVec3(runningMax, bins[i].boundsMax);
                runningCount += bins[i].count;
            }
            rightMin[i] = runningMin;
            rightMax[i] = runningMax;
            rightCount[i] = runningCount;
        }

        for (int i = 0; i < binCount - 1; ++i) {
            int leftNum = leftCount[i];
            int rightNum = rightCount[i + 1];
            if (leftNum == 0 || rightNum == 0) {
                continue;
            }

            Real leftArea = surfaceArea(leftMin[i], leftMax[i]);
            Real rightArea = surfaceArea(rightMin[i + 1], rightMax[i + 1]);
            Real cost = (leftArea * leftNum + rightArea * rightNum) / parentArea;

            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplit = i;
            }
        }
    }

    if (bestAxis < 0) {
        return false;
    }

    if (bestCost >= static_cast<Real>(triCount)) {
        return false;
    }

    Real extent = centroidExtent[bestAxis];
    if (extent <= Real(1.0e-6)) {
        return false;
    }

    Real splitPos = centroidMin[bestAxis] + extent * (static_cast<Real>(bestSplit + 1) /
                                                      static_cast<Real>(binCount));

    auto midIter = std::partition(
        triangles.begin() + start,
        triangles.begin() + end,
        [bestAxis, splitPos](const Triangle& tri) {
            vec3 c = triangleCentroid(tri);
            return c[bestAxis] < splitPos;
        });

    mid = static_cast<int>(midIter - triangles.begin());
    return mid > start && mid < end;
}

int allocateNode(buildContext& ctx) {
    int index = ctx.nodeCounter->fetch_add(1);
    return index;
}

int buildNode(buildContext& ctx, int start, int end, int depth) {
    std::vector<Triangle>& triangles = *ctx.triangles;
    std::vector<BvhNode>& nodes = *ctx.nodes;

    vec3 boundsMin;
    vec3 boundsMax;
    computeBounds(triangles, start, end, boundsMin, boundsMax);

    int nodeIndex = allocateNode(ctx);
    BvhNode& node = nodes[static_cast<std::size_t>(nodeIndex)];
    node.boundsMin = boundsMin;
    node.boundsMax = boundsMax;
    node.left = -1;
    node.right = -1;
    node.firstTri = start;
    node.triCount = end - start;

    if (node.triCount <= ctx.options.leafSize) {
        return nodeIndex;
    }

    int mid = -1;
    bool splitOk = false;
    if (ctx.options.algorithm == bvhBuildAlgorithm::Sah) {
        splitOk = partitionSah(triangles, start, end, ctx.options.binCount, boundsMin, boundsMax, mid);
    }
    if (!splitOk) {
        splitOk = partitionSimple(triangles, start, end, mid);
    }
    if (!splitOk) {
        return nodeIndex;
    }

    int leftIndex = -1;
    int rightIndex = -1;
    bool shouldParallel = ctx.options.enableParallel
        && (end - start) >= ctx.options.minTrianglesForParallel
        && depth < ctx.options.maxParallelDepth;

    if (shouldParallel) {
#ifdef _OPENMP
        #pragma omp task shared(leftIndex)
        leftIndex = buildNode(ctx, start, mid, depth + 1);
        #pragma omp task shared(rightIndex)
        rightIndex = buildNode(ctx, mid, end, depth + 1);
        #pragma omp taskwait
#else
        std::thread leftThread([&]() { leftIndex = buildNode(ctx, start, mid, depth + 1); });
        rightIndex = buildNode(ctx, mid, end, depth + 1);
        leftThread.join();
#endif
    } else {
        leftIndex = buildNode(ctx, start, mid, depth + 1);
        rightIndex = buildNode(ctx, mid, end, depth + 1);
    }

    node.left = leftIndex;
    node.right = rightIndex;
    node.firstTri = 0;
    node.triCount = 0;

    return nodeIndex;
}
}

bvhBuildResult buildBvh(std::vector<Triangle> triangles, const bvhBuildOptions& options) {
    bvhBuildResult result;
    result.triangles = std::move(triangles);

    if (result.triangles.empty()) {
        return result;
    }

    bvhBuildOptions buildOptions = options;
    if (buildOptions.leafSize < 1) {
        buildOptions.leafSize = 1;
    }
    if (buildOptions.binCount < 2) {
        buildOptions.binCount = 2;
    }

    std::size_t maxNodes = result.triangles.size() * 2 - 1;
    result.nodes.resize(maxNodes);
    std::atomic<int> nodeCounter{0};

    buildContext ctx{
        &result.triangles,
        &result.nodes,
        &nodeCounter,
        buildOptions
    };

#ifdef _OPENMP
    if (buildOptions.enableParallel) {
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                result.rootIndex = buildNode(ctx, 0, static_cast<int>(result.triangles.size()), 0);
            }
        }
    } else {
        result.rootIndex = buildNode(ctx, 0, static_cast<int>(result.triangles.size()), 0);
    }
#else
    result.rootIndex = buildNode(ctx, 0, static_cast<int>(result.triangles.size()), 0);
#endif

    result.nodes.resize(static_cast<std::size_t>(nodeCounter.load()));
    return result;
}
