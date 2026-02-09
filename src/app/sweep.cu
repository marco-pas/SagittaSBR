// MPI-parallelized simulation sweep

#include "sim/sweep.hpp"

#include <mpi.h>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "app/print.hpp"
#include "cuda/cudaUtils.cuh"
#include "cuda/gpuConfig.cuh"
#include "cuda/poKernels.cuh"
#include "cuda/rtKernels.cuh"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <nvtx3/nvToolsExt.h>
#endif

// Structure to hold results from each iteration
struct IterationResult {
    int globalIdx;
    Real thetaDeg;
    Real phiDeg;
    Real sigma;
    Real rcsDbsm;
};

sweepResults runSweep(const simulationConfig& config, deviceBuffers& buffers,
                      const bvhGpuData& bvhData, std::ostream& outFile) {

    // Initialize MPI
    int rank, nProcs;


    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

#ifndef USE_HIP
    if (rank == 0) nvtxRangePushA("runSweep");
#endif
    
    // Physical params
    const Real c0 = REAL_CONST(299792458.0);
    const Real lambda = c0 / config.freq;
    const Real k = REAL_CONST(2.0) * Real(M_PI) / lambda;
    const Real rayArea = (config.gridSize * config.gridSize) / (config.nx * config.ny);
    const int nRays = config.nx * config.ny;
    const Real distanceOffset = REAL_CONST(10.0) * config.gridSize;

    // Print simulation parameters (rank 0 only)
    if (rank == 0) {
        printSeparator("SIMULATION PARAMETERS");
        printKv("MPI Processes", nProcs);
        printKv("Frequency", config.freq / 1e9, "GHz");
        printKv("Wavelength", lambda * 1000, "mm");
        printKv("Wave Number k", k, "rad/m");
        printKv("Illumination Area", config.gridSize * config.gridSize, "m²");
        printKv("Grid Size", std::to_string(config.nx) + "x" + std::to_string(config.ny));
        printKv("Total Rays", nRays);
        printKv("Ray Density", nRays / (config.gridSize * config.gridSize), "rays/m²");
        printKv("Ray Area", rayArea, "m²");
        printKv("Phi Range", std::to_string(config.phiStart) + " to " + std::to_string(config.phiEnd));
        printKv("Theta Range", std::to_string(config.thetaStart) + " to " + std::to_string(config.thetaEnd));
        printKv("Show GPU info", std::to_string(config.showInfoGPU));
        printKv("Show Hit info", std::to_string(config.showHitStats));
        printEndSeparator();

        // Physics validation
        Real raySpacing = realSqrt(rayArea);
        Real samplingRatio = raySpacing / lambda;

        if (samplingRatio > REAL_CONST(0.2)) {
            printSeparator("WARNING");
            std::cerr << "│ Physics is under-resolved at your frequency!\n";
            printKv("Ray spacing / λ", samplingRatio, " (< 0.1 needed)");
            printKv("Undersampled by", samplingRatio / REAL_CONST(0.1), "times");
            
            std::cerr << "│\n│ Solutions: \n";
            printKv("  a. Increase n_x, n_y by factor", realSqrt(samplingRatio / REAL_CONST(0.1)));
            printKv("  b. Reduce frequency below", (config.freq * REAL_CONST(0.1) / samplingRatio) / REAL_CONST(1e9), "GHz");
            std::cerr << "│   c. Reduce illumination area (if possible)\n";
            printEndSeparator();
        }
    }

    // GPU configuration
    dim3 rtThreads(GPU_RT_BLOCK_X, GPU_RT_BLOCK_Y);
    dim3 rtBlocks(GPU_RT_GRID_X(config.nx), GPU_RT_GRID_Y(config.ny));

    // Create GPU events for timing
    cudaEvent_t rtKernelStart, rtKernelStop;
    cudaEvent_t poKernelStart, poKernelStop;
    cudaEvent_t memsetStart, memsetStop;
    cudaEvent_t memcpyStart, memcpyStop;
    
    checkCudaErrors(cudaEventCreate(&rtKernelStart));
    checkCudaErrors(cudaEventCreate(&rtKernelStop));
    checkCudaErrors(cudaEventCreate(&poKernelStart));
    checkCudaErrors(cudaEventCreate(&poKernelStop));
    checkCudaErrors(cudaEventCreate(&memsetStart));
    checkCudaErrors(cudaEventCreate(&memsetStop));
    checkCudaErrors(cudaEventCreate(&memcpyStart));
    checkCudaErrors(cudaEventCreate(&memcpyStop));

    // Timing accumulators (per rank)
    double totalRtKernelMs = 0.0;
    double totalPoKernelMs = 0.0;
    double totalMemsetMs = 0.0;
    double totalMemcpyMs = 0.0;
    double totalFileWriteMs = 0.0;
    double totalStatsMs = 0.0;

    // Calculate total iterations and work distribution
    const int totalIterations = config.thetaSamples * config.phiSamples;
    const int iterPerProc = totalIterations / nProcs;
    const int remainder = totalIterations % nProcs;
    
    // Each rank gets iterPerProc iterations, with first 'remainder' ranks getting one extra
    const int myStart = rank * iterPerProc + std::min(rank, remainder);
    const int myCount = iterPerProc + (rank < remainder ? 1 : 0);
    const int myEnd = myStart + myCount;

    if (rank == 0) {
        printSeparator("EXECUTING SWEEP");
        std::cerr << "│ Total iterations: " << totalIterations << "\n";
        std::cerr << "│ Distributed across " << nProcs << " MPI processes\n";
        printEndSeparator();
    }

    clock_t totalSweepStart = clock();
    
    // Storage for this rank's results
    std::vector<IterationResult> localResults;
    localResults.reserve(myCount);

    // Single flattened loop - each rank processes its assigned subset

    // Start of Theta + Phi loop with MPI

    for (int globalIdx = myStart; globalIdx < myEnd; ++globalIdx) {
        // Convert flat index back to theta and phi indices
        const int t = globalIdx / config.phiSamples;
        const int p = globalIdx % config.phiSamples;

        // Calculate theta and phi values
        Real thetaDeg = config.thetaStart + (config.thetaSamples > 1
            ? t * (config.thetaEnd - config.thetaStart) / (config.thetaSamples - 1)
            : 0);
        Real thetaRad = thetaDeg * Real(M_PI) / REAL_CONST(180.0);

        Real phiDeg = config.phiStart + (config.phiSamples > 1
            ? p * (config.phiEnd - config.phiStart) / (config.phiSamples - 1)
            : 0);
        Real phiRad = phiDeg * Real(M_PI) / REAL_CONST(180.0);

        // Calculate ray direction and basis vectors
        vec3 dir(
            realSin(thetaRad) * realCos(phiRad), 
            realSin(thetaRad) * realSin(phiRad), 
            realCos(thetaRad)
        );

        vec3 rayDir = -dir;

        vec3 up(0, 1, 0);
        if (realAbs(dir.y()) > REAL_CONST(0.99)) {
            up = vec3(1, 0, 0);
        }

        vec3 uVec = unitVector(cross(up, dir)) * config.gridSize;
        vec3 vVec = unitVector(cross(dir, uVec)) * config.gridSize;
        vec3 llc = (dir * distanceOffset) - REAL_CONST(0.5) * uVec - REAL_CONST(0.5) * vVec;

        // ========== TIMING: Memset ==========
        checkCudaErrors(cudaEventRecord(memsetStart));
        checkCudaErrors(cudaMemset(buffers.accum, 0, sizeof(cuRealComplex)));
        checkCudaErrors(cudaEventRecord(memsetStop));

        // ========== TIMING: Ray tracing kernel ==========
        checkCudaErrors(cudaEventRecord(rtKernelStart));
        launchRaysMultiBounce<<<rtBlocks, rtThreads>>>(
            buffers.hitPos, buffers.hitNormal, buffers.lastDir,
            buffers.hitDist, buffers.hitCount,
            config.nx, config.ny, llc, uVec, vVec, rayDir,
            bvhData.triangles, bvhData.nodes, bvhData.rootIndex,
            config.maxBounces);
        checkCudaErrors(cudaEventRecord(rtKernelStop));

        // ========== TIMING: PO integration kernel ==========
        checkCudaErrors(cudaEventRecord(poKernelStart));
        integratePoMultiBounce<<<GPU_GRID_SIZE(nRays, GPU_PO_BLOCK_SIZE), GPU_PO_BLOCK_SIZE>>>(
            buffers.hitPos, buffers.hitNormal, buffers.lastDir,
            buffers.hitDist, buffers.hitCount,
            nRays, k, rayDir, rayArea, buffers.accum, config.reflectionConst);
        checkCudaErrors(cudaEventRecord(poKernelStop));

        // ========== TIMING: Device to host memcpy ==========
        checkCudaErrors(cudaEventRecord(memcpyStart));
        checkCudaErrors(cudaMemcpy(buffers.accumHost, buffers.accum, 
                                   sizeof(cuRealComplex), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaEventRecord(memcpyStop));
        checkCudaErrors(cudaEventSynchronize(memcpyStop));

        // Get timing results
        float rtKernelMs = 0.0f, poKernelMs = 0.0f, memsetMs = 0.0f, memcpyMs = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&memsetMs, memsetStart, memsetStop));
        checkCudaErrors(cudaEventElapsedTime(&rtKernelMs, rtKernelStart, rtKernelStop));
        checkCudaErrors(cudaEventElapsedTime(&poKernelMs, poKernelStart, poKernelStop));
        checkCudaErrors(cudaEventElapsedTime(&memcpyMs, memcpyStart, memcpyStop));

        totalMemsetMs += memsetMs;
        totalRtKernelMs += rtKernelMs;
        totalPoKernelMs += poKernelMs;
        totalMemcpyMs += memcpyMs;

        // Calculate RCS
        Real sigma = REAL_CONST(4.0) * Real(M_PI) * (buffers.accumHost->x * buffers.accumHost->x
                     + buffers.accumHost->y * buffers.accumHost->y);
        Real rcsDbsm = REAL_CONST(10.0) * realLog10(realFmax(sigma, REAL_CONST(1e-10)));

        // Store result
        localResults.push_back({globalIdx, thetaDeg, phiDeg, sigma, rcsDbsm});

        float totalIterMs = memsetMs + rtKernelMs + poKernelMs + memcpyMs;

        // Print progress with rank info
        if (config.showHitStats) {
            clock_t statsStart = clock();
            
            checkCudaErrors(cudaMemcpy(buffers.hitCountHost, buffers.hitCount,
                                       nRays * sizeof(int), cudaMemcpyDeviceToHost));
            
            int maxHits = 0;
            int zeroCount = 0;
            long long sumHits = 0;
            int nonZeroCount = 0;

            for (int n = 0; n < nRays; n++) {
                int val = buffers.hitCountHost[n];
                if (val == 0) {
                    zeroCount++;
                } else {
                    if (val > maxHits) maxHits = val;
                    sumHits += val;
                    nonZeroCount++;
                }
            }

            clock_t statsEnd = clock();
            double statsMs = double(statsEnd - statsStart) / CLOCKS_PER_SEC * 1000.0;
            totalStatsMs += statsMs;

            Real avgHits = (nonZeroCount > 0) ? static_cast<Real>(sumHits) / nonZeroCount : REAL_CONST(0.0);

            std::cerr << "[R" << rank << ":" << std::setw(3) << globalIdx + 1 << "/"
                << totalIterations << "]"
                << " │ Θ: " << std::setw(3) << static_cast<int>(thetaDeg) << "°"
                << " │ Φ: " << std::setw(3) << static_cast<int>(phiDeg) << "°"
                << " │ RT: " << std::fixed << std::setprecision(2) << rtKernelMs << "ms"
                << " │ PO: " << poKernelMs << "ms"
                << " │ Tot: " << totalIterMs << "ms"
                << " │ Hit: " << std::setprecision(0) << 100.0f - (100.0f * zeroCount / nRays) << "%"
                << " │ Bounces: " << std::setprecision(2) << avgHits
                << "\n";
        } else {
            std::cerr << "[R" << rank << ":" << std::setw(3) << globalIdx + 1 << "/"
                << totalIterations << "]"
                << " │ Θ: " << std::setw(3) << static_cast<int>(thetaDeg) << "°"
                << " │ Φ: " << std::setw(3) << static_cast<int>(phiDeg) << "°"
                << " │ RT: " << std::fixed << std::setprecision(2) << rtKernelMs << "ms"
                << " │ PO: " << poKernelMs << "ms"
                << " │ Total: " << totalIterMs << "ms"
                << "\n";
        }
    }

    // Cleanup GPU events
    checkCudaErrors(cudaEventDestroy(rtKernelStart));
    checkCudaErrors(cudaEventDestroy(rtKernelStop));
    checkCudaErrors(cudaEventDestroy(poKernelStart));
    checkCudaErrors(cudaEventDestroy(poKernelStop));
    checkCudaErrors(cudaEventDestroy(memsetStart));
    checkCudaErrors(cudaEventDestroy(memsetStop));
    checkCudaErrors(cudaEventDestroy(memcpyStart));
    checkCudaErrors(cudaEventDestroy(memcpyStop));

    clock_t totalSweepEnd = clock();
    double localTime = double(totalSweepEnd - totalSweepStart) / CLOCKS_PER_SEC;

    // Gather all results to rank 0
    int localCount = localResults.size();
    std::vector<int> recvCounts(nProcs);
    MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(nProcs);
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < nProcs; ++i) {
            displs[i] = displs[i-1] + recvCounts[i-1];
        }
    }

    // Prepare data for gathering (convert struct to arrays)
    std::vector<int> localGlobalIdx(localCount);
    std::vector<Real> localTheta(localCount), localPhi(localCount);
    std::vector<Real> localSigma(localCount), localRcs(localCount);
    
    for (int i = 0; i < localCount; ++i) {
        localGlobalIdx[i] = localResults[i].globalIdx;
        localTheta[i] = localResults[i].thetaDeg;
        localPhi[i] = localResults[i].phiDeg;
        localSigma[i] = localResults[i].sigma;
        localRcs[i] = localResults[i].rcsDbsm;
    }

    std::vector<int> allGlobalIdx;
    std::vector<Real> allTheta, allPhi, allSigma, allRcs;
    
    if (rank == 0) {
        allGlobalIdx.resize(totalIterations);
        allTheta.resize(totalIterations);
        allPhi.resize(totalIterations);
        allSigma.resize(totalIterations);
        allRcs.resize(totalIterations);
    }

    MPI_Gatherv(localGlobalIdx.data(), localCount, MPI_INT,
                allGlobalIdx.data(), recvCounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localTheta.data(), localCount, REAL_MPI_TYPE,
                allTheta.data(), recvCounts.data(), displs.data(), REAL_MPI_TYPE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localPhi.data(), localCount, REAL_MPI_TYPE,
                allPhi.data(), recvCounts.data(), displs.data(), REAL_MPI_TYPE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localSigma.data(), localCount, REAL_MPI_TYPE,
                allSigma.data(), recvCounts.data(), displs.data(), REAL_MPI_TYPE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localRcs.data(), localCount, REAL_MPI_TYPE,
                allRcs.data(), recvCounts.data(), displs.data(), REAL_MPI_TYPE,
                0, MPI_COMM_WORLD);

    // Gather timing statistics
    double globalRtKernelMs, globalPoKernelMs, globalMemsetMs, globalMemcpyMs;
    double globalStatsMs, globalMaxTime;
    
    MPI_Reduce(&totalRtKernelMs, &globalRtKernelMs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalPoKernelMs, &globalPoKernelMs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalMemsetMs, &globalMemsetMs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalMemcpyMs, &globalMemcpyMs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalStatsMs, &globalStatsMs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localTime, &globalMaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Rank 0: Sort results by global index and write to file
    if (rank == 0) {
        clock_t fileWriteStart = clock();
        
        // Create index mapping for sorting
        std::vector<std::pair<int, int>> indexMap(totalIterations);
        for (int i = 0; i < totalIterations; ++i) {
            indexMap[i] = {allGlobalIdx[i], i};
        }
        std::sort(indexMap.begin(), indexMap.end());

        // Write results in correct order
        for (int i = 0; i < totalIterations; ++i) {
            int idx = indexMap[i].second;
            outFile << allTheta[idx] << "," << allPhi[idx] << "," 
                    << allSigma[idx] << "," << allRcs[idx] << "\n";
        }
        
        clock_t fileWriteEnd = clock();
        totalFileWriteMs = double(fileWriteEnd - fileWriteStart) / CLOCKS_PER_SEC * 1000.0;

        printEndSeparator();

        // Print performance summary
        printSeparator("SWEEP PERFORMANCE BREAKDOWN");
        printKv("Ray Tracing Kernel (total)", globalRtKernelMs, "ms");
        printKv("PO Integration Kernel (total)", globalPoKernelMs, "ms");
        printKv("Memset (total)", globalMemsetMs, "ms");
        printKv("Memcpy D2H (total)", globalMemcpyMs, "ms");
        printKv("File Writes (total)", totalFileWriteMs, "ms");
        if (config.showHitStats) {
            printKv("Stats Computation (total)", globalStatsMs, "ms");
        }
        printKv("─────────────────", "");
        printKv("GPU Time (kernels only)", globalRtKernelMs + globalPoKernelMs, "ms");
        printKv("Total Sweep Wall Time", globalMaxTime * 1000.0, "ms");
        printEndSeparator();

        printSeparator("PER-ITERATION AVERAGES");
        printKv("Ray Tracing Kernel", globalRtKernelMs / totalIterations, "ms");
        printKv("PO Integration Kernel", globalPoKernelMs / totalIterations, "ms");
        printKv("Memset", globalMemsetMs / totalIterations, "ms");
        printKv("Memcpy D2H", globalMemcpyMs / totalIterations, "ms");
        printKv("File Write", totalFileWriteMs / totalIterations, "ms");
        printKv("─────────────────", "");
        printKv("Total Points", totalIterations);
        printKv("Average Time/Point", globalMaxTime * 1000.0 / totalIterations, "ms");
        printKv("Speedup (vs sequential)", (globalRtKernelMs + globalPoKernelMs) / (globalMaxTime * 1000.0), "x");
        printEndSeparator();
    }

#ifndef USE_HIP
    if (rank == 0) nvtxRangePop();
#endif

    return sweepResults{totalIterations, globalMaxTime};
}