#ifndef SCENE_BVH_BUILDER_HPP
#define SCENE_BVH_BUILDER_HPP

#include <vector>

#include "scene/bvh.h"

enum class bvhBuildAlgorithm {
    Simple,
    Sah
};

struct bvhBuildOptions {
    int leafSize = 4;
    int binCount = 16;
    int maxParallelDepth = 6;
    int minTrianglesForParallel = 4096;
    bool enableParallel = true;
    bvhBuildAlgorithm algorithm = bvhBuildAlgorithm::Sah;
};

struct bvhBuildResult {
    std::vector<Triangle> triangles;
    std::vector<BvhNode> nodes;
    int rootIndex = 0;
};

bvhBuildResult buildBvh(std::vector<Triangle> triangles,
                        const bvhBuildOptions& options = {});

#endif
