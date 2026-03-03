// ------------- MPI-parallelized simulation sweep ------------- //

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

    // ----------------------------------------------------------------------
    // @@ Multi-stream pipeline with double-buffered accumulators
    //
    // Two streams allow overlapping computation with data movement:
    //   computeStream  – memset, RT kernel, PO kernel (all serialized here
    //                     because RT and PO share the hit-data buffers)
    //   transferStream – D2H copy of the accumulator result
    //
    // Two accum buffers (device + host) let iteration N+1's kernels run on
    // computeStream while iteration N's result is still being transferred
    // and processed on the CPU.
    //
    // Timeline (simplified):
    //   computeStream:  |--RT[0]--PO[0]--|--RT[1]--PO[1]--|--RT[2]--...
    //   transferStream:          |D2H[0]--|       |D2H[1]--|
    //   CPU:                       |proc[0]|        |proc[1]|
    //
    // The poComplete event ensures transferStream waits for PO to finish
    // before starting the D2H copy. The computeStream serializes RT→PO
    // naturally, and cannot start RT[N+1] until PO[N] is done (same stream).
    // ----------------------------------------------------------------------

    constexpr int NUM_BUFS = 2;

    // Streams
    cudaStream_t computeStream, transferStream;
    checkCudaErrors(cudaStreamCreate(&computeStream));
    checkCudaErrors(cudaStreamCreate(&transferStream));

    // Double-buffered accumulators (device + pinned host)
    cuRealComplex* accumDev[NUM_BUFS];
    cuRealComplex* accumHost[NUM_BUFS];
    accumDev[0]  = buffers.accum;      // reuse existing allocation
    accumHost[0] = buffers.accumHost;
    checkCudaErrors(cudaMalloc(&accumDev[1], sizeof(cuRealComplex)));
    checkCudaErrors(cudaMallocHost(&accumHost[1], sizeof(cuRealComplex)));

    // Synchronization: signals when PO kernel is done on computeStream
    cudaEvent_t poComplete;
    checkCudaErrors(cudaEventCreate(&poComplete));

    // Per-buffer event: marks when the iteration's RT kernel STARTS on
    // computeStream.  Used by the showHitStats path to make the next RT
    // kernel wait until the CPU has finished reading the shared hitCount
    // buffer from the previous iteration.
    cudaEvent_t hitCountCopied;
    checkCudaErrors(cudaEventCreateWithFlags(&hitCountCopied, cudaEventDisableTiming));

    // Per-buffer timing events (one set per buffer so we can query while
    // the other buffer's events are still being recorded)
    cudaEvent_t rtKernelStart[NUM_BUFS], rtKernelStop[NUM_BUFS];
    cudaEvent_t poKernelStart[NUM_BUFS], poKernelStop[NUM_BUFS];
    cudaEvent_t memsetStart[NUM_BUFS],   memsetStop[NUM_BUFS];
    cudaEvent_t memcpyStart[NUM_BUFS],   memcpyStop[NUM_BUFS];

    for (int b = 0; b < NUM_BUFS; ++b) {
        checkCudaErrors(cudaEventCreate(&rtKernelStart[b]));
        checkCudaErrors(cudaEventCreate(&rtKernelStop[b]));
        checkCudaErrors(cudaEventCreate(&poKernelStart[b]));
        checkCudaErrors(cudaEventCreate(&poKernelStop[b]));
        checkCudaErrors(cudaEventCreate(&memsetStart[b]));
        checkCudaErrors(cudaEventCreate(&memsetStop[b]));
        checkCudaErrors(cudaEventCreate(&memcpyStart[b]));
        checkCudaErrors(cudaEventCreate(&memcpyStop[b]));
    }

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

    // How often to print progress (1 = every iteration, 100 = every 100th, etc.)
    // When showHitStats is enabled, non-printed iterations skip the expensive
    // 320 MB hitCount D2H + 80M-element CPU loop, saving ~20 ms/iter.
    constexpr int PRINT_INTERVAL = 100;

    // Counter for completed iterations on this rank (used for print interval)
    int completedCount = 0;

    // Whether the next RT kernel must wait for a hitCount D2H on the CPU side.
    // Set to true after processResult records hitCountCopied on the host;
    // consumed by the next kernel-submit block which inserts a
    // cudaStreamWaitEvent before the RT kernel launch.
    bool needHitCountFence = false;

    // ------------------------------------------------------------------------
    // Helper lambda: process a completed iteration's result.
    // Called after the transfer stream has been synchronized for that buffer.
    // Collects timing + RCS unconditionally; prints only every PRINT_INTERVAL.
    // ------------------------------------------------------------------------
    auto processResult = [&](int buf, int prevGlobalIdx, Real prevThetaDeg,
                             Real prevPhiDeg) {
        // Query timing events (all guaranteed complete after transfer sync)
        float rtMs = 0.0f, poMs = 0.0f, msMs = 0.0f, cpMs = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msMs, memsetStart[buf], memsetStop[buf]));
        checkCudaErrors(cudaEventElapsedTime(&rtMs, rtKernelStart[buf], rtKernelStop[buf]));
        checkCudaErrors(cudaEventElapsedTime(&poMs, poKernelStart[buf], poKernelStop[buf]));
        checkCudaErrors(cudaEventElapsedTime(&cpMs, memcpyStart[buf], memcpyStop[buf]));

        totalMemsetMs   += msMs;
        totalRtKernelMs += rtMs;
        totalPoKernelMs += poMs;
        totalMemcpyMs   += cpMs;

        // Calculate RCS from the completed buffer
        Real sigma = REAL_CONST(4.0) * Real(M_PI)
                     * (accumHost[buf]->x * accumHost[buf]->x
                      + accumHost[buf]->y * accumHost[buf]->y);
        Real rcsDbsm = REAL_CONST(10.0) * realLog10(realFmax(sigma, REAL_CONST(1e-10)));

        localResults.push_back({prevGlobalIdx, prevThetaDeg, prevPhiDeg, sigma, rcsDbsm});

        ++completedCount;

        // Only print every PRINT_INTERVAL iterations (and always print the last)
        bool isLast = (prevGlobalIdx == myEnd - 1);
        if (completedCount % PRINT_INTERVAL != 0 && !isLast) return;

        float totalIterMs = msMs + rtMs + poMs + cpMs;

        // Print progress — this runs while the GPU is busy with the next iteration
        if (config.showHitStats) {
            clock_t statsStart = clock();

            // hitCount is a shared buffer written by the RT kernel.
            // The next iteration's RT kernel has already been submitted to
            // computeStream but is blocked by a cudaStreamWaitEvent on
            // hitCountCopied.  We read hitCount here, then record the event
            // to release the pending RT kernel.
            //
            // The previous iteration's PO kernel (which was the last writer
            // after RT on computeStream) is guaranteed complete because the
            // transferStream D2H (which we just synchronized) waited on the
            // poComplete event.
            checkCudaErrors(cudaMemcpy(buffers.hitCountHost, buffers.hitCount,
                                       nRays * sizeof(int), cudaMemcpyDeviceToHost));

            // Release the pending RT kernel on computeStream
            checkCudaErrors(cudaEventRecord(hitCountCopied));
            needHitCountFence = false;

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

            std::cerr << "[R" << rank << ":" << std::setw(3) << prevGlobalIdx + 1 << "/"
                << totalIterations << "]"
                << " │ Θ: " << std::setw(3) << static_cast<int>(prevThetaDeg) << "°"
                << " │ Φ: " << std::setw(3) << static_cast<int>(prevPhiDeg) << "°"
                << " │ RT: " << std::fixed << std::setprecision(2) << rtMs << "ms"
                << " │ PO: " << poMs << "ms"
                << " │ Tot: " << totalIterMs << "ms"
                << " │ Hit: " << std::setprecision(0) << 100.0f - (100.0f * zeroCount / nRays) << "%"
                << " │ Bounces: " << std::setprecision(2) << avgHits
                << "\n";
        } else {
            std::cerr << "[R" << rank << ":" << std::setw(3) << prevGlobalIdx + 1 << "/"
                << totalIterations << "]"
                << " │ Θ: " << std::setw(3) << static_cast<int>(prevThetaDeg) << "°"
                << " │ Φ: " << std::setw(3) << static_cast<int>(prevPhiDeg) << "°"
                << " │ RT: " << std::fixed << std::setprecision(2) << rtMs << "ms"
                << " │ PO: " << poMs << "ms"
                << " │ Total: " << totalIterMs << "ms"
                << "\n";
        }
    };

    // ------------------------------------------------------------------------
    // @@ Main sweep loop — pipelined
    //
    // Iteration timeline (GPU busy while CPU processes previous result):
    //
    // computeStream:  |memset+RT[0]+PO[0]|memset+RT[1]+PO[1]|memset+RT[2]...
    // transferStream:           |D2H[0]--|        |D2H[1]--|
    // CPU:            angles[0] submit[0] angles[1] submit[1] drain[0] ...
    //                                                         ↑ GPU is running
    //
    // Key: kernel submission happens BEFORE draining the previous result,
    // so the GPU starts new work immediately while the CPU catches up.
    //
    // When showHitStats is enabled and a print is due, the next RT kernel
    // is held back via cudaStreamWaitEvent(hitCountCopied) so the CPU can
    // safely read the shared hitCount buffer.  The memset for that iteration
    // still runs (it doesn't touch hitCount), limiting the stall.
    // ------------------------------------------------------------------------
    bool hasPending = false;
    int pendingBuf = 0;
    int pendingGlobalIdx = 0;
    Real pendingThetaDeg = REAL_CONST(0.0);
    Real pendingPhiDeg   = REAL_CONST(0.0);

    for (int globalIdx = myStart; globalIdx < myEnd; ++globalIdx) {
        int buf = (globalIdx - myStart) % NUM_BUFS;

        // 1. CPU: compute angles (pure CPU, overlaps with GPU tail)
        const int t = globalIdx / config.phiSamples;
        const int p = globalIdx % config.phiSamples;

        Real thetaDeg = config.thetaStart + (config.thetaSamples > 1
            ? t * (config.thetaEnd - config.thetaStart) / (config.thetaSamples - 1)
            : 0);
        Real thetaRad = thetaDeg * Real(M_PI) / REAL_CONST(180.0);

        Real phiDeg = config.phiStart + (config.phiSamples > 1
            ? p * (config.phiEnd - config.phiStart) / (config.phiSamples - 1)
            : 0);
        Real phiRad = phiDeg * Real(M_PI) / REAL_CONST(180.0);

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

        // 2. Submit GPU work on computeStream (GPU starts ASAP)

        // Memset this buffer's accumulator (safe even if hitCount fence is
        // pending — memset touches accumDev, not hitCount)
        checkCudaErrors(cudaEventRecord(memsetStart[buf], computeStream));
        checkCudaErrors(cudaMemsetAsync(accumDev[buf], 0, sizeof(cuRealComplex), computeStream));
        checkCudaErrors(cudaEventRecord(memsetStop[buf], computeStream));

        // If showHitStats printed on the previous drain, the RT kernel must
        // wait until the CPU has finished the synchronous hitCount D2H copy
        // so we don't overwrite the buffer mid-read.
        if (needHitCountFence) {
            checkCudaErrors(cudaStreamWaitEvent(computeStream, hitCountCopied));
            needHitCountFence = false;
        }

        // @@ Ray tracing kernel
        checkCudaErrors(cudaEventRecord(rtKernelStart[buf], computeStream)); 
        launchRaysMultiBounce<<<rtBlocks, rtThreads, 0, computeStream>>>(                                               // Ray launch
            buffers.hitNormal, buffers.lastDir,
            buffers.hitDist, buffers.hitCount,
            config.nx, config.ny, llc, uVec, vVec, rayDir,
            bvhData.triangles, bvhData.nodes, bvhData.rootIndex,
            config.maxBounces);
        checkCudaErrors(cudaEventRecord(rtKernelStop[buf], computeStream));

        // @@ PO integration kernel — writes to accumDev[buf]
        checkCudaErrors(cudaEventRecord(poKernelStart[buf], computeStream));
        integratePoMultiBounce<<<GPU_GRID_SIZE(nRays, GPU_PO_BLOCK_SIZE), GPU_PO_BLOCK_SIZE, 0, computeStream>>>(       // PO Integral
            buffers.hitNormal, buffers.lastDir,
            buffers.hitDist, buffers.hitCount,
            nRays, k, rayDir, rayArea, accumDev[buf], config.reflectionConst);
        checkCudaErrors(cudaEventRecord(poKernelStop[buf], computeStream));

        // Signal that PO is done so transferStream can start the D2H copy
        checkCudaErrors(cudaEventRecord(poComplete, computeStream));

        // D2H copy on transferStream (overlaps with next compute)
        checkCudaErrors(cudaStreamWaitEvent(transferStream, poComplete));
        checkCudaErrors(cudaEventRecord(memcpyStart[buf], transferStream));
        checkCudaErrors(cudaMemcpyAsync(accumHost[buf], accumDev[buf],
                                        sizeof(cuRealComplex), cudaMemcpyDeviceToHost,
                                        transferStream));
        checkCudaErrors(cudaEventRecord(memcpyStop[buf], transferStream));

        // 3. Drain previous result (GPU is now busy with this iter)
        if (hasPending) {
            checkCudaErrors(cudaStreamSynchronize(transferStream));

            // Determine if processResult will need to read hitCount.
            // If so, set the fence flag so the NEXT iteration's RT kernel
            // waits for the read to complete.
            int nextCompleted = completedCount + 1;
            bool willPrint = (nextCompleted % PRINT_INTERVAL == 0)
                          || (pendingGlobalIdx == myEnd - 1);
            if (config.showHitStats && willPrint) {
                needHitCountFence = true;
            }

            processResult(pendingBuf, pendingGlobalIdx, pendingThetaDeg, pendingPhiDeg);
            hasPending = false;
        }

        // 4. Mark this iteration as pending
        hasPending = true;
        pendingBuf = buf;
        pendingGlobalIdx = globalIdx;
        pendingThetaDeg = thetaDeg;
        pendingPhiDeg = phiDeg;
    }

    // Drain the last pending iteration
    if (hasPending) {
        checkCudaErrors(cudaStreamSynchronize(transferStream));
        processResult(pendingBuf, pendingGlobalIdx, pendingThetaDeg, pendingPhiDeg);
    }

    // Cleanup streams, events, and extra buffer
    checkCudaErrors(cudaStreamDestroy(computeStream));
    checkCudaErrors(cudaStreamDestroy(transferStream));
    checkCudaErrors(cudaEventDestroy(poComplete));
    checkCudaErrors(cudaEventDestroy(hitCountCopied));
    for (int b = 0; b < NUM_BUFS; ++b) {
        checkCudaErrors(cudaEventDestroy(rtKernelStart[b]));
        checkCudaErrors(cudaEventDestroy(rtKernelStop[b]));
        checkCudaErrors(cudaEventDestroy(poKernelStart[b]));
        checkCudaErrors(cudaEventDestroy(poKernelStop[b]));
        checkCudaErrors(cudaEventDestroy(memsetStart[b]));
        checkCudaErrors(cudaEventDestroy(memsetStop[b]));
        checkCudaErrors(cudaEventDestroy(memcpyStart[b]));
        checkCudaErrors(cudaEventDestroy(memcpyStop[b]));
    }
    checkCudaErrors(cudaFree(accumDev[1]));
    checkCudaErrors(cudaFreeHost(accumHost[1]));

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

// ------------------------------------------------------------- //