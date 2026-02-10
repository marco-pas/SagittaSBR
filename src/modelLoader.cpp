// ModelLoader.cpp

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyOBJLoader.hpp"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tinyGLTF.hpp"

#include "modelLoader.hpp"

#include <unordered_map>
#include <cstring>
#include <algorithm>
#include <cctype>

namespace modelLoader {

static std::string toLower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

static bool hasExtension(const std::string& filePath, const std::string& extLower) {
    std::string lower = toLower(filePath);
    if (lower.size() < extLower.size()) {
        return false;
    }
    return lower.compare(lower.size() - extLower.size(), extLower.size(), extLower) == 0;
}


// OBJ loading (tinyobjloader)

namespace {

struct ObjVertexKey {
    int posIndex;
    int normalIndex;
    int texcoordIndex;

    bool operator==(const ObjVertexKey& other) const noexcept {
        return posIndex == other.posIndex &&
               normalIndex == other.normalIndex &&
               texcoordIndex == other.texcoordIndex;
    }
};

struct ObjVertexKeyHash {
    std::size_t operator()(const ObjVertexKey& k) const noexcept {
        std::size_t h1 = std::hash<int>{}(k.posIndex);
        std::size_t h2 = std::hash<int>{}(k.normalIndex);
        std::size_t h3 = std::hash<int>{}(k.texcoordIndex);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

bool loadObjInternal(
    const std::string& filePath,
    MeshData& outMesh,
    std::string& outError
) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    std::string baseDir;
    {
        std::size_t slashPos = filePath.find_last_of("/\\");
        if (slashPos != std::string::npos) {
            baseDir = filePath.substr(0, slashPos + 1);
        }
    }

    bool ret = tinyobj::LoadObj(
        &attrib,
        &shapes,
        &materials,
        &warn,
        &err,
        filePath.c_str(),
        baseDir.empty() ? nullptr : baseDir.c_str(),
        true,
        false
    );

    if (!warn.empty()) {
        outError += "tinyobj warning: " + warn + "\n";
    }
    if (!ret || !err.empty()) {
        outError += "tinyobj error: " + err + "\n";
        return false;
    }

    outMesh.positions.clear();
    outMesh.normals.clear();
    outMesh.texcoords.clear();
    outMesh.indices.clear();

    std::unordered_map<ObjVertexKey, std::uint32_t, ObjVertexKeyHash> vertexMap;
    vertexMap.reserve(1024);

    auto addVertex = [&](const tinyobj::index_t& idx) -> std::uint32_t {
        ObjVertexKey key{ idx.vertex_index, idx.normal_index, idx.texcoord_index };

        auto it = vertexMap.find(key);
        if (it != vertexMap.end()) {
            return it->second;
        }

        std::uint32_t newIndex = static_cast<std::uint32_t>(outMesh.positions.size() / 3);

        if (idx.vertex_index >= 0) {
            int vIdx = 3 * idx.vertex_index;
            outMesh.positions.push_back(attrib.vertices[vIdx + 0]);
            outMesh.positions.push_back(attrib.vertices[vIdx + 1]);
            outMesh.positions.push_back(attrib.vertices[vIdx + 2]);
        } else {
            outMesh.positions.push_back(0.0f);
            outMesh.positions.push_back(0.0f);
            outMesh.positions.push_back(0.0f);
        }

        if (!attrib.normals.empty()) {
            if (idx.normal_index >= 0) {
                int nIdx = 3 * idx.normal_index;
                outMesh.normals.push_back(attrib.normals[nIdx + 0]);
                outMesh.normals.push_back(attrib.normals[nIdx + 1]);
                outMesh.normals.push_back(attrib.normals[nIdx + 2]);
            } else {
                outMesh.normals.push_back(0.0f);
                outMesh.normals.push_back(0.0f);
                outMesh.normals.push_back(0.0f);
            }
        }

        if (!attrib.texcoords.empty()) {
            if (idx.texcoord_index >= 0) {
                int tIdx = 2 * idx.texcoord_index;
                outMesh.texcoords.push_back(attrib.texcoords[tIdx + 0]);
                outMesh.texcoords.push_back(attrib.texcoords[tIdx + 1]);
            } else {
                outMesh.texcoords.push_back(0.0f);
                outMesh.texcoords.push_back(0.0f);
            }
        }

        vertexMap.emplace(key, newIndex);
        return newIndex;
    };

    for (const auto& shape : shapes) {
        std::size_t indexOffset = 0;
        for (std::size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];

            if (fv != 3) {
                indexOffset += fv;
                continue;
            }

            std::uint32_t idx0 = addVertex(shape.mesh.indices[indexOffset + 0]);
            std::uint32_t idx1 = addVertex(shape.mesh.indices[indexOffset + 1]);
            std::uint32_t idx2 = addVertex(shape.mesh.indices[indexOffset + 2]);

            outMesh.indices.push_back(idx0);
            outMesh.indices.push_back(idx1);
            outMesh.indices.push_back(idx2);

            indexOffset += fv;
        }
    }

    return true;
}

} // anonymous namespace (OBJ)

// GLTF loading (tinygltf)

namespace {

void copyAccessorToFloatArray(
    const tinygltf::Model& model,
    int accessorIndex,
    int expectedNumComponents,
    std::vector<float>& outData
) {
    if (accessorIndex < 0) {
        outData.clear();
        return;
    }

    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

    if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        outData.clear();
        return;
    }

    std::size_t numComponents =
        tinygltf::GetNumComponentsInType(accessor.type);
    std::size_t componentSize =
        tinygltf::GetComponentSizeInBytes(accessor.componentType);

    std::size_t stride = bufferView.byteStride;
    if (stride == 0) {
        stride = numComponents * componentSize;
    }

    const unsigned char* basePtr =
        buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

    std::size_t count = accessor.count;
    outData.resize(count * expectedNumComponents);

    for (std::size_t i = 0; i < count; ++i) {
        const float* src =
            reinterpret_cast<const float*>(basePtr + i * stride);

        for (int c = 0; c < expectedNumComponents; ++c) {
            if (c < static_cast<int>(numComponents)) {
                outData[i * expectedNumComponents + c] = src[c];
            } else {
                outData[i * expectedNumComponents + c] = 0.0f;
            }
        }
    }
}

void copyIndices(
    const tinygltf::Model& model,
    int accessorIndex,
    std::vector<std::uint32_t>& outIndices
) {
    if (accessorIndex < 0) {
        outIndices.clear();
        return;
    }

    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

    std::size_t componentSize =
        tinygltf::GetComponentSizeInBytes(accessor.componentType);
    std::size_t stride = bufferView.byteStride;
    if (stride == 0) {
        stride = componentSize;
    }

    const unsigned char* basePtr =
        buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

    std::size_t count = accessor.count;
    outIndices.resize(count);

    for (std::size_t i = 0; i < count; ++i) {
        const unsigned char* ptr = basePtr + i * stride;
        std::uint32_t index = 0;

        switch (accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            index = static_cast<std::uint32_t>(*reinterpret_cast<const std::uint8_t*>(ptr));
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            index = static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(ptr));
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            index = *reinterpret_cast<const std::uint32_t*>(ptr);
            break;
        default:
            index = 0;
            break;
        }

        outIndices[i] = index;
    }
}

bool loadGltfInternal(
    const std::string& filePath,
    MeshData& outMesh,
    std::string& outError
) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string warn;
    std::string err;

    bool isBinary = hasExtension(filePath, ".glb");

    bool success = false;
    if (isBinary) {
        success = loader.LoadBinaryFromFile(&model, &err, &warn, filePath);
    } else {
        success = loader.LoadASCIIFromFile(&model, &err, &warn, filePath);
    }

    if (!warn.empty()) {
        outError += "tinygltf warning: " + warn + "\n";
    }
    if (!success || !err.empty()) {
        outError += "tinygltf error: " + err + "\n";
        return false;
    }

    outMesh.positions.clear();
    outMesh.normals.clear();
    outMesh.texcoords.clear();
    outMesh.indices.clear();

    std::vector<float> localPos;
    std::vector<float> localNorm;
    std::vector<float> localUv;
    std::vector<std::uint32_t> localIdx;

    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }

            auto itPos = prim.attributes.find("POSITION");
            if (itPos == prim.attributes.end()) {
                continue;
            }

            int posAccessorIndex = itPos->second;
            copyAccessorToFloatArray(model, posAccessorIndex, 3, localPos);
            std::size_t vertexCount = localPos.size() / 3;
            if (vertexCount == 0) {
                continue;
            }

            auto itNorm = prim.attributes.find("NORMAL");
            if (itNorm != prim.attributes.end()) {
                copyAccessorToFloatArray(model, itNorm->second, 3, localNorm);
            } else {
                localNorm.clear();
            }

            auto itUv = prim.attributes.find("TEXCOORD_0");
            if (itUv != prim.attributes.end()) {
                copyAccessorToFloatArray(model, itUv->second, 2, localUv);
            } else {
                localUv.clear();
            }

            if (prim.indices >= 0) {
                copyIndices(model, prim.indices, localIdx);
            } else {
                localIdx.resize(vertexCount);
                for (std::size_t i = 0; i < vertexCount; ++i) {
                    localIdx[i] = static_cast<std::uint32_t>(i);
                }
            }

            std::uint32_t baseVertex =
                static_cast<std::uint32_t>(outMesh.positions.size() / 3);

            outMesh.positions.insert(
                outMesh.positions.end(),
                localPos.begin(), localPos.end()
            );

            if (!localNorm.empty()) {
                if (outMesh.normals.empty()) {
                    outMesh.normals.resize(baseVertex * 3, 0.0f);
                }
                outMesh.normals.insert(
                    outMesh.normals.end(),
                    localNorm.begin(), localNorm.end()
                );
            } else {
                if (!outMesh.normals.empty()) {
                    outMesh.normals.resize(
                        (baseVertex + vertexCount) * 3,
                        0.0f
                    );
                }
            }

            if (!localUv.empty()) {
                if (outMesh.texcoords.empty()) {
                    outMesh.texcoords.resize(baseVertex * 2, 0.0f);
                }
                outMesh.texcoords.insert(
                    outMesh.texcoords.end(),
                    localUv.begin(), localUv.end()
                );
            } else {
                if (!outMesh.texcoords.empty()) {
                    outMesh.texcoords.resize(
                        (baseVertex + vertexCount) * 2,
                        0.0f
                    );
                }
            }

            for (std::uint32_t idx : localIdx) {
                outMesh.indices.push_back(baseVertex + idx);
            }
        }
    }

    return true;
}

} // anonymous namespace (GLTF)

// Public API

bool loadModel(
    const std::string& filePath,
    MeshData& outMesh,
    std::string& outError,
    FileType fileType
) {
    outError.clear();

    FileType type = fileType;
    if (type == FileType::Auto) {
        if (hasExtension(filePath, ".obj")) {
            type = FileType::Obj;
        } else if (hasExtension(filePath, ".gltf") || hasExtension(filePath, ".glb")) {
            type = FileType::Gltf;
        } else {
            outError = "Unsupported file extension for \"" + filePath + "\"";
            return false;
        }
    }

    switch (type) {
    case FileType::Obj:
        return loadObjInternal(filePath, outMesh, outError);
    case FileType::Gltf:
        return loadGltfInternal(filePath, outMesh, outError);
    default:
        outError = "Unknown FileType";
        return false;
    }
}
} // namespace modelLoader