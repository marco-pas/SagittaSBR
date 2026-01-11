// ModelLoader.hpp
#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace modelLoader {

struct MeshData {
    std::vector<float> positions;      // [x0,y0,z0, x1,y1,z1, ...]
    std::vector<float> normals;        // [nx0,ny0,nz0, ...] (may be empty)
    std::vector<float> texcoords;      // [u0,v0, u1,v1, ...] (may be empty)
    std::vector<std::uint32_t> indices; // [i0,i1,i2, ...] (3 per triangle)
};

enum class FileType {
    Auto,  // Infer from file extension
    Obj,
    Gltf
};

bool loadModel(
    const std::string& filePath,
    MeshData& outMesh,
    std::string& outError,
    FileType fileType = FileType::Auto
);

} // namespace modelLoader