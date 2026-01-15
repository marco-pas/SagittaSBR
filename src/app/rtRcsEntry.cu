#include "app/rtRcsEntry.hpp"

#include <fstream>
#include <cstdint>
#include <cctype>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "app/config.hpp"
#include "app/deviceBuffers.hpp"
#include "app/print.hpp"
#include "cuda/cudaUtils.cuh"
#include "model/modelLoader.hpp"
#include "scene/bvh.h"
#include "scene/bvhBuilder.hpp"
#include "sim/sweep.hpp"

namespace {
std::string toLowerCopy(const std::string& value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower;
}

bool endsWith(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool resolveModelFileType(const std::string& path, modelLoader::FileType& outType) {
    std::string lower = toLowerCopy(path);
    if (endsWith(lower, ".obj")) {
        outType = modelLoader::FileType::Obj;
        return true;
    }
    if (endsWith(lower, ".gltf") || endsWith(lower, ".glb")) {
        outType = modelLoader::FileType::Gltf;
        return true;
    }
    return false;
}
}

int runRcsApp(int argc, char** argv) {
    std::cerr << "╔═══════════════════════════════════════════════════╗\n";
    std::cerr << "║  MONOSTATIC RCS SWEEP - SHOOTING & BOUNCING RAYS  ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════╝\n";

    printGpuInfo();

    simulationConfig config = loadConfig("config.txt", argc, argv);
    int rayCount = config.nx * config.ny;

    if (config.modelPath.empty()) {
        std::cerr << "Error: --model is required for simulation runs.\n";
        return 1;
    }

    modelLoader::MeshData mesh;
    std::string modelError;
    modelLoader::FileType modelType = modelLoader::FileType::Auto;

    if (!resolveModelFileType(config.modelPath, modelType)) {
        std::cerr << "Error: Unsupported model extension for '"
                  << config.modelPath << "'.\n";
        return 1;
    }

    if (!modelLoader::loadModel(config.modelPath, mesh, modelError, modelType)) {
        std::cerr << "Error: Failed to load model '" << config.modelPath << "'.\n";
        if (!modelError.empty()) {
            std::cerr << modelError;
        }
        return 1;
    }

    if (mesh.indices.empty() || mesh.positions.empty() || (mesh.indices.size() % 3) != 0) {
        std::cerr << "Error: Model '" << config.modelPath
                  << "' contains no valid triangles.\n";
        return 1;
    }

    std::size_t vertexCount = mesh.positions.size() / 3;
    std::size_t triangleCount = mesh.indices.size() / 3;

    std::vector<Triangle> triangles;
    triangles.reserve(triangleCount);
    for (std::size_t i = 0; i < triangleCount; ++i) {
        std::size_t base = i * 3;
        std::uint32_t i0 = mesh.indices[base + 0];
        std::uint32_t i1 = mesh.indices[base + 1];
        std::uint32_t i2 = mesh.indices[base + 2];

        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) {
            std::cerr << "Error: Model '" << config.modelPath
                      << "' contains out-of-range indices.\n";
            return 1;
        }

        std::size_t v0 = i0 * 3;
        std::size_t v1 = i1 * 3;
        std::size_t v2 = i2 * 3;

        Triangle tri;
        tri.v0 = vec3(mesh.positions[v0 + 0] * config.modelScale,
                      mesh.positions[v0 + 1] * config.modelScale,
                      mesh.positions[v0 + 2] * config.modelScale);
        tri.v1 = vec3(mesh.positions[v1 + 0] * config.modelScale,
                      mesh.positions[v1 + 1] * config.modelScale,
                      mesh.positions[v1 + 2] * config.modelScale);
        tri.v2 = vec3(mesh.positions[v2 + 0] * config.modelScale,
                      mesh.positions[v2 + 1] * config.modelScale,
                      mesh.positions[v2 + 2] * config.modelScale);

        triangles.push_back(tri);
    }

    bvhBuildResult bvh = buildBvh(std::move(triangles));

    printSeparator("MEMORY ALLOCATION");
    deviceBuffers buffers;
    allocateDeviceBuffers(buffers, rayCount);

    bvhGpuData bvhData;
    bvhData.triangleCount = static_cast<int>(bvh.triangles.size());
    bvhData.nodeCount = static_cast<int>(bvh.nodes.size());
    bvhData.rootIndex = bvh.rootIndex;

    checkCudaErrors(cudaMalloc(&bvhData.triangles, bvhData.triangleCount * sizeof(Triangle)));
    checkCudaErrors(cudaMemcpy(bvhData.triangles, bvh.triangles.data(),
                               bvhData.triangleCount * sizeof(Triangle),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&bvhData.nodes, bvhData.nodeCount * sizeof(BvhNode)));
    checkCudaErrors(cudaMemcpy(bvhData.nodes, bvh.nodes.data(),
                               bvhData.nodeCount * sizeof(BvhNode),
                               cudaMemcpyHostToDevice));
    std::cerr << "│  BVH buffers and world setup complete.\n";
    printEndSeparator();

    std::ofstream outFile("rcs_results.csv");
    outFile << "# Frequency: " << config.freq << "\n# Grid: " << config.nx << "x" << config.ny << "\n";
    outFile << "# GridSize: " << config.gridSize << "\n# ThetaSamples: " << config.thetaSamples << "\n";
    outFile << "# PhiSamples: " << config.phiSamples << "\n" << "theta,phi,rcs_m2,rcs_dbsm\n";

    runSweep(config, buffers, bvhData, outFile);

    printSeparator("CLEANUP");
    freeDeviceBuffers(buffers);
    checkCudaErrors(cudaFree(bvhData.triangles));
    checkCudaErrors(cudaFree(bvhData.nodes));
    std::cerr << "│  Device memory released. Success.\n";
    printEndSeparator();

    return 0;
}
