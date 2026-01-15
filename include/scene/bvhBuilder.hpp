#ifndef SCENE_BVH_BUILDER_HPP
#define SCENE_BVH_BUILDER_HPP

#include <vector>

#include "scene/bvh.h"

struct bvhBuildResult {
    std::vector<Triangle> triangles;
    std::vector<BvhNode> nodes;
    int rootIndex = 0;
};

bvhBuildResult buildBvh(std::vector<Triangle> triangles, int leafSize = 4);

#endif
