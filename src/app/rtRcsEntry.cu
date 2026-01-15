#include "app/rtRcsEntry.hpp"

#include <fstream>
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
#include "scene/world.cuh"
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

    modelLoader::MeshData mesh;
    std::string modelError;
    bool useMesh = false;
    int triangleCount = 0;
    std::size_t vertexCount = 0;
    modelLoader::FileType modelType = modelLoader::FileType::Auto;

    if (!config.modelPath.empty()) {
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

        vertexCount = mesh.positions.size() / 3;
        triangleCount = static_cast<int>(mesh.indices.size() / 3);
        useMesh = true;
    }

    printSeparator("MEMORY ALLOCATION");
    deviceBuffers buffers;
    allocateDeviceBuffers(buffers, rayCount);

    vec3* deviceVertices = nullptr;
    int* deviceIndices = nullptr;
    hitable** deviceList = nullptr;
    hitable** deviceWorld = nullptr;
    checkCudaErrors(cudaMalloc(&deviceWorld, sizeof(hitable*)));

    if (useMesh) {
        checkCudaErrors(cudaMallocManaged(&deviceVertices, vertexCount * sizeof(vec3)));
        checkCudaErrors(cudaMallocManaged(&deviceIndices, mesh.indices.size() * sizeof(int)));
        for (std::size_t i = 0; i < vertexCount; ++i) {
            std::size_t base = i * 3;
            deviceVertices[i] = vec3(mesh.positions[base + 0] * config.modelScale,
                                     mesh.positions[base + 1] * config.modelScale,
                                     mesh.positions[base + 2] * config.modelScale);
        }
        for (std::size_t i = 0; i < mesh.indices.size(); ++i) {
            deviceIndices[i] = static_cast<int>(mesh.indices[i]);
        }

        checkCudaErrors(cudaMalloc(&deviceList, triangleCount * sizeof(hitable*)));
        createWorldFromMesh<<<1, 1>>>(
            deviceList, deviceWorld, deviceVertices, deviceIndices, triangleCount);
    } else {
        checkCudaErrors(cudaMalloc(&deviceList, sizeof(hitable*)));
        createWorld<<<1, 1>>>(deviceList, deviceWorld);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "│  Managed memory and world setup complete.\n";
    printEndSeparator();

    std::ofstream outFile("rcs_results.csv");
    outFile << "# Frequency: " << config.freq << "\n# Grid: " << config.nx << "x" << config.ny << "\n";
    outFile << "# GridSize: " << config.gridSize << "\n# ThetaSamples: " << config.thetaSamples << "\n";
    outFile << "# PhiSamples: " << config.phiSamples << "\n" << "theta,phi,rcs_m2,rcs_dbsm\n";

    runSweep(config, buffers, deviceWorld, outFile);

    printSeparator("CLEANUP");
    if (useMesh) {
        freeWorldFromMesh<<<1, 1>>>(deviceList, deviceWorld, triangleCount);
    } else {
        freeWorld<<<1, 1>>>(deviceList, deviceWorld);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    freeDeviceBuffers(buffers);
    checkCudaErrors(cudaFree(deviceList));
    checkCudaErrors(cudaFree(deviceWorld));
    if (useMesh) {
        checkCudaErrors(cudaFree(deviceVertices));
        checkCudaErrors(cudaFree(deviceIndices));
    }
    std::cerr << "│  Device memory released. Success.\n";
    printEndSeparator();

    return 0;
}
