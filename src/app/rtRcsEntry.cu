// actual entry point

#include "app/rtRcsEntry.hpp"

#include <mpi.h>
#include <fstream>
#include <cstdint>
#include <cctype>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "app/config.hpp"
#include "app/deviceBuffers.hpp"
#include "app/print.hpp"
#include "cuda/cudaUtils.cuh"
#include "model/modelLoader.hpp"
#include "scene/bvh.h"
#include "scene/bvhBuilder.hpp"
#include "sim/sweep.hpp"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <nvtx3/nvToolsExt.h>
#endif

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

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cerr << "╔═══════════════════════════════════════════════════╗\n";
        std::cerr << "║  SagittaSBR  >>---->  MONOSTATIC RCS CALCULATION  ║\n";
        std::cerr << "╚═══════════════════════════════════════════════════╝\n";
    }

    // @@ timing: Total application
    clock_t appStartTime = clock();

    simulationConfig config = loadConfig("config.txt", argc, argv);

    if (config.showInfoGPU && rank == 0) printGpuInfo();

    int rayCount = config.nx * config.ny;

    if (config.modelPath.empty()) {
        std::cerr << "Error: --model is required for simulation runs.\n";
        return 1;
    }

#ifndef USE_HIP
    nvtxRangePushA("Read Mesh");
#endif

    // @@ timing: Model load
    clock_t modelLoadStart = clock();

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
        tri.v0 = vec3(static_cast<Real>(mesh.positions[v0 + 0]) * config.modelScale,
                      static_cast<Real>(mesh.positions[v0 + 1]) * config.modelScale,
                      static_cast<Real>(mesh.positions[v0 + 2]) * config.modelScale);
        tri.v1 = vec3(static_cast<Real>(mesh.positions[v1 + 0]) * config.modelScale,
                      static_cast<Real>(mesh.positions[v1 + 1]) * config.modelScale,
                      static_cast<Real>(mesh.positions[v1 + 2]) * config.modelScale);
        tri.v2 = vec3(static_cast<Real>(mesh.positions[v2 + 0]) * config.modelScale,
                      static_cast<Real>(mesh.positions[v2 + 1]) * config.modelScale,
                      static_cast<Real>(mesh.positions[v2 + 2]) * config.modelScale);

        triangles.push_back(tri);
    }

    clock_t modelLoadEnd = clock();
    double modelLoadTime = double(modelLoadEnd - modelLoadStart) / CLOCKS_PER_SEC;

#ifndef USE_HIP
    nvtxRangePop();
#endif

    // @@ timing: BVH construction
    clock_t bvhBuildStart = clock();

    bvhBuildOptions buildOptions;
    // Switch buildOptions.algorithm to bvhBuildAlgorithm::Simple for the median splitter.

#ifndef USE_HIP
    nvtxRangePushA("build BVH");
#endif
    bvhBuildResult bvh = buildBvh(std::move(triangles), buildOptions);
#ifndef USE_HIP
    nvtxRangePop();
#endif

    clock_t bvhBuildEnd = clock();
    double bvhBuildTime = double(bvhBuildEnd - bvhBuildStart) / CLOCKS_PER_SEC;

    // @@ timing: Memory allocation
    clock_t memAllocStart = clock();

    if (rank == 0) printSeparator("MEMORY ALLOCATION");
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
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    clock_t memAllocEnd = clock();
    double memAllocTime = double(memAllocEnd - memAllocStart) / CLOCKS_PER_SEC;

    if (rank == 0) {
        std::cerr << "│  BVH buffers and world setup complete.\n";
        printEndSeparator();
    }

    // Print initialization timing
    if (rank == 0) {
        printSeparator("INITIALIZATION @@ timing");
        printKv("Model Loading", modelLoadTime * 1000.0, "ms");
        printKv("BVH Construction", bvhBuildTime * 1000.0, "ms");
        printKv("GPU Memory Alloc", memAllocTime * 1000.0, "ms");
        printKv("Total Init Time", (modelLoadTime + bvhBuildTime + memAllocTime) * 1000.0, "ms");
        printKv("Triangle Count", static_cast<int>(triangleCount));
        printKv("BVH Node Count", bvhData.nodeCount);
        printEndSeparator();
    }

    // Print estimated memory footprint
    if (rank == 0) {
        printSeparator("MEMORY FOOTPRINT ESTIMATE");

        const std::size_t sizeofReal = sizeof(Real);
        const std::size_t sizeofVec3 = sizeof(vec3);
        const std::size_t sizeofTriangle = sizeof(Triangle);
        const std::size_t sizeofBvhNode = sizeof(BvhNode);
        const std::size_t sizeofComplex = sizeof(cuRealComplex);

        // GPU device memory 
        std::size_t gpuHitNormal   = rayCount * sizeofVec3;
        std::size_t gpuLastDir     = rayCount * sizeofVec3;
        std::size_t gpuHitDist     = rayCount * sizeofReal;
        std::size_t gpuHitCount    = rayCount * sizeof(int);
        std::size_t gpuAccum       = 2 * sizeofComplex;       // double-buffered in sweep
        std::size_t gpuBvhTris     = bvhData.triangleCount * sizeofTriangle;
        std::size_t gpuBvhNodes    = bvhData.nodeCount * sizeofBvhNode;

        std::size_t gpuRayBuffers  = gpuHitNormal + gpuLastDir + gpuHitDist + gpuHitCount;
        std::size_t gpuBvhTotal    = gpuBvhTris + gpuBvhNodes;
        std::size_t gpuTotal       = gpuRayBuffers + gpuBvhTotal + gpuAccum;

        // Host pinned memory 
        std::size_t pinnedHitCount = rayCount * sizeof(int);
        std::size_t pinnedAccum    = 2 * sizeofComplex;       // double-buffered in sweep
        std::size_t pinnedTotal    = pinnedHitCount + pinnedAccum;

        // Host heap (persistent during sweep)
        // BVH build result vectors are moved/consumed, mesh is transient.
        // localResults vector in sweep (per iteration: 5 values)
        int totalIterations = config.thetaSamples * config.phiSamples;
        std::size_t hostSweepResults = totalIterations * (sizeof(int) + 4 * sizeofReal);
        // MPI gather buffers (rank 0 only)
        std::size_t hostGatherBufs  = totalIterations * (sizeof(int) + 4 * sizeofReal);
        std::size_t hostTotal       = hostSweepResults + hostGatherBufs;

        std::size_t grandTotal = gpuTotal + pinnedTotal + hostTotal;

        auto toMB = [](std::size_t bytes) -> double {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        };

        printKv("Precision", sizeofReal == 8 ? "double (64-bit)" : "float (32-bit)");
        printKv("sizeof(Real)", sizeofReal, "B");
        printKv("sizeof(vec3)", sizeofVec3, "B");
        printKv("sizeof(Triangle)", sizeofTriangle, "B");
        printKv("sizeof(BvhNode)", sizeofBvhNode, "B");

        std::cerr << "│\n";
        std::cerr << "│  ── GPU Device Memory ──\n";
        printKv("  hitNormal", toMB(gpuHitNormal), "MB");
        printKv("  lastDir", toMB(gpuLastDir), "MB");
        printKv("  hitDist", toMB(gpuHitDist), "MB");
        printKv("  hitCount", toMB(gpuHitCount), "MB");
        printKv("  accum (x2)", toMB(gpuAccum), "MB");
        printKv("  BVH triangles", toMB(gpuBvhTris), "MB");
        printKv("  BVH nodes", toMB(gpuBvhNodes), "MB");
        printKv("  ─── Subtotal GPU", toMB(gpuTotal), "MB");

        std::cerr << "│\n";
        std::cerr << "│  ── Host Pinned Memory ──\n";
        printKv("  hitCountHost", toMB(pinnedHitCount), "MB");
        printKv("  accumHost (x2)", toMB(pinnedAccum), "MB");
        printKv("  ─── Subtotal Pinned", toMB(pinnedTotal), "MB");

        std::cerr << "│\n";
        std::cerr << "│  ── Host Heap (sweep) ──\n";
        printKv("  localResults", toMB(hostSweepResults), "MB");
        printKv("  MPI gather bufs", toMB(hostGatherBufs), "MB");
        printKv("  ─── Subtotal Host", toMB(hostTotal), "MB");

        std::cerr << "│\n";
        printKv("  ═══ TOTAL ESTIMATED", toMB(grandTotal), "MB");
        printEndSeparator();
    }

    // @@ timing: File open (rank 0 only)
    clock_t fileOpenStart = clock();
    std::ofstream outFile;
    if (rank == 0) {
        outFile.open("rcs_results.csv");
        outFile << "# Frequency: " << config.freq << "\n# Grid: " << config.nx << "x" << config.ny << "\n";
        outFile << "# GridSize: " << config.gridSize << "\n# ThetaSamples: " << config.thetaSamples << "\n";
        outFile << "# PhiSamples: " << config.phiSamples << "\n" << "theta,phi,rcs_m2,rcs_dbsm\n";
    }
    clock_t fileOpenEnd = clock();
    double fileOpenTime = double(fileOpenEnd - fileOpenStart) / CLOCKS_PER_SEC;

    // Run sweep (with detailed timing inside)
    // Note: only rank 0 writes to outFile; other ranks pass an unopened stream
    sweepResults results = runSweep(config, buffers, bvhData, outFile);

    // @@ timing: File close
    clock_t fileCloseStart = clock();
    if (rank == 0) outFile.close();
    clock_t fileCloseEnd = clock();
    double fileCloseTime = double(fileCloseEnd - fileCloseStart) / CLOCKS_PER_SEC;

    // @@ timing: Cleanup
    clock_t cleanupStart = clock();
    
    if (rank == 0) printSeparator("CLEANUP");
    freeDeviceBuffers(buffers);
    checkCudaErrors(cudaFree(bvhData.triangles));
    checkCudaErrors(cudaFree(bvhData.nodes));
    if (rank == 0) {
        std::cerr << "│  Device memory released. Success.\n";
        printEndSeparator();
    }
    
    clock_t cleanupEnd = clock();
    double cleanupTime = double(cleanupEnd - cleanupStart) / CLOCKS_PER_SEC;

    // Total application time
    clock_t appEndTime = clock();
    double totalAppTime = double(appEndTime - appStartTime) / CLOCKS_PER_SEC;

    // Print final timing summary
    if (rank == 0) {
        printSeparator("TOTAL @@ timing BREAKDOWN");
        printKv("Model Loading", modelLoadTime * 1000.0, "ms");
        printKv("BVH Construction", bvhBuildTime * 1000.0, "ms");
        printKv("GPU Memory Alloc", memAllocTime * 1000.0, "ms");
        printKv("File Open/Header", fileOpenTime * 1000.0, "ms");
        printKv("Sweep Execution", results.totalTimeSeconds * 1000.0, "ms");
        printKv("File Close", fileCloseTime * 1000.0, "ms");
        printKv("Cleanup", cleanupTime * 1000.0, "ms");
        printKv("─────────────────", "");
        printKv("Total App Time", totalAppTime * 1000.0, "ms");
        printEndSeparator();
    }

    return 0;
}