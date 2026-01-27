// actual simulation sweep

#include "sim/sweep.hpp"

#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

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

sweepResults runSweep(const simulationConfig& config, deviceBuffers& buffers,
                      const bvhGpuData& bvhData, std::ostream& outFile) {

#ifndef USE_HIP
    nvtxRangePushA("runSweep");
#endif
    
    // physical params
    const float c0 = 299792458.0f;
    const float lambda = c0 / config.freq;
    const float k = 2.0f * M_PI / lambda;
    const float rayArea = (config.gridSize * config.gridSize) / (config.nx * config.ny);
    const int nRays = config.nx * config.ny;
    const float distanceOffset = 10.0f * config.gridSize;  // = 10.0f; // this has to be fixed!

    // simulation params
    printSeparator("SIMULATION PARAMETERS");
    printKv("Frequency", config.freq / 1e9, "GHz");
    printKv("Wavelength", lambda * 1000, "mm");
    printKv("Wave Number k", k, "rad/m");
    printKv("Illumination Area", config.gridSize * config.gridSize,
            "m²");
    printKv("Grid Size", std::to_string(config.nx) + "x" + std::to_string(config.ny));
    printKv("Total Rays", nRays);
    printKv("Ray Density", nRays / (config.gridSize * config.gridSize),
            "rays/m²");
    printKv("Ray Area", rayArea, "m²");
    printKv("Phi Range", std::to_string(config.phiStart) + " to " + std::to_string(config.phiEnd));
    printKv("Theta Range", std::to_string(config.thetaStart) + " to " + std::to_string(config.thetaEnd));
    printKv("Show GPU info", std::to_string(config.showInfoGPU));
    printKv("Show Hit info", std::to_string(config.showHitStats));
    printEndSeparator();

    // Make sure physics is well represented
    float raySpacing = std::sqrt(rayArea);
    float samplingRatio = raySpacing / lambda;  // Should be < 0.1 ideally

    if (samplingRatio > 0.2f) {  // or 0.1f for stricter
        printSeparator("WARNING");
        std::cerr << "│ Physics is under-resolved at your frequency!\n";
        printKv("Ray spacing / λ", samplingRatio, " (< 0.1 needed)");
        printKv("Undersampled by", samplingRatio / 0.1f, "times");
        
        std::cerr << "│\n│ Solutions: \n";
        printKv("  a. Increase n_x, n_y by factor", std::sqrt(samplingRatio / 0.1f));
        printKv("  b. Reduce frequency below", (config.freq * 0.1f / samplingRatio) / 1e9, "GHz");
        std::cerr << "│   c. Reduce illumination area (if possible)\n";
        printEndSeparator();
    }


    dim3 threads(config.tpbx, config.tpby);
    dim3 blocks(
        (config.nx + config.tpbx - 1) / config.tpbx, 
        (config.ny + config.tpby - 1) / config.tpby  // BUG FIX: was missing -1
    );

    // ========== Create GPU events for detailed timing ==========
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

    // Accumulators for timing statistics
    double totalRtKernelMs = 0.0;
    double totalPoKernelMs = 0.0;
    double totalMemsetMs = 0.0;
    double totalMemcpyMs = 0.0;
    double totalFileWriteMs = 0.0;
    double totalStatsMs = 0.0;

    printSeparator("EXECUTING SWEEP");
    clock_t totalSweepStart = clock();
    int totalIterations = 0;

    for (int t = 0; t < config.thetaSamples; ++t) {
        float thetaDeg = config.thetaStart + (config.thetaSamples > 1
            ? t * (config.thetaEnd - config.thetaStart) / (config.thetaSamples - 1)
            : 0);
        float thetaRad = thetaDeg * M_PI / 180.0f;

        for (int p = 0; p < config.phiSamples; ++p) {
            float phiDeg = config.phiStart + (config.phiSamples > 1
                ? p * (config.phiEnd - config.phiStart) / (config.phiSamples - 1)
                : 0);
            float phiRad = phiDeg * M_PI / 180.0f;

            vec3 dir(
                sinf(thetaRad) * cosf(phiRad), 
                sinf(thetaRad) * sinf(phiRad), 
                cosf(thetaRad)
            );

            vec3 rayDir = -dir; // invert as ray travels towards the center

            vec3 up(0, 1, 0);
            if (fabsf(dir.y()) > 0.99f) { // avoids NaNs
                up = vec3(1, 0, 0);
            }

            vec3 uVec = unitVector(cross(up, dir)) * config.gridSize;
            vec3 vVec = unitVector(cross(dir, uVec)) * config.gridSize;
            vec3 llc = (dir * distanceOffset) - 0.5f * uVec - 0.5f * vVec;

            // ========== TIMING: Memset ==========
            checkCudaErrors(cudaEventRecord(memsetStart));
            checkCudaErrors(cudaMemset(buffers.accum, 0, sizeof(cuFloatComplex)));
            checkCudaErrors(cudaEventRecord(memsetStop));

            // ========== TIMING: Ray tracing kernel ==========
            checkCudaErrors(cudaEventRecord(rtKernelStart));
            launchRaysMultiBounce<<<blocks, threads>>>(
                buffers.hitPos, buffers.hitNormal, buffers.hitDist, buffers.hitCount,
                config.nx, config.ny, llc, uVec, vVec, rayDir,
                bvhData.triangles, bvhData.nodes, bvhData.rootIndex,
                config.maxBounces);
            checkCudaErrors(cudaEventRecord(rtKernelStop));

            // ========== TIMING: PO integration kernel ==========
            checkCudaErrors(cudaEventRecord(poKernelStart));
            integratePoMultiBounce<<<GPU_GRID_SIZE(nRays, GPU_BLOCK_SIZE_1D), GPU_BLOCK_SIZE_1D>>>(
                buffers.hitPos, buffers.hitNormal, buffers.hitDist, buffers.hitCount,
                nRays, k, rayDir, rayArea, buffers.accum, config.reflectionConst);
            checkCudaErrors(cudaEventRecord(poKernelStop));

            // ========== TIMING: Device to host memcpy ==========
            checkCudaErrors(cudaEventRecord(memcpyStart));
            checkCudaErrors(cudaMemcpy(buffers.accumHost, buffers.accum, 
                                       sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
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

            float sigma = 4.0f * M_PI * (buffers.accumHost->x * buffers.accumHost->x
                         + buffers.accumHost->y * buffers.accumHost->y);
            float rcsDbsm = 10.0f * log10f(fmaxf(sigma, 1e-10f));

            // ========== TIMING: File write ==========
            clock_t fileWriteStart = clock();
            outFile << thetaDeg << "," << phiDeg << "," << sigma << "," << rcsDbsm << "\n";
            clock_t fileWriteEnd = clock();
            double fileWriteMs = double(fileWriteEnd - fileWriteStart) / CLOCKS_PER_SEC * 1000.0;
            totalFileWriteMs += fileWriteMs;

            float totalIterMs = memsetMs + rtKernelMs + poKernelMs + memcpyMs + fileWriteMs;

            if (config.showHitStats) {
                // ========== TIMING: Stats computation ==========
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

                float avgHits = (nonZeroCount > 0) ? static_cast<float>(sumHits) / nonZeroCount : 0.0f;

                std::cerr << "[" << std::setw(3) << totalIterations + 1 << "/"
                    << config.phiSamples * config.thetaSamples << "]"
                    << " │ Θ: " << std::setw(3) << static_cast<int>(thetaDeg) << "°"
                    << " │ Φ: " << std::setw(3) << static_cast<int>(phiDeg) << "°"
                    << " │ RT: " << std::fixed << std::setprecision(2) << rtKernelMs << "ms"
                    << " │ PO: " << poKernelMs << "ms"
                    << " │ Tot: " << totalIterMs << "ms"
                    << " │ Hit: " << std::setprecision(0) << 100.0f - (100.0f * zeroCount / nRays) << "%"
                    << " │ Bounces: " << std::setprecision(2) << avgHits
                    << "\n";
            } else {
                std::cerr << "[" << std::setw(3) << totalIterations + 1 << "/"
                    << config.phiSamples * config.thetaSamples << "]"
                    << " │ Θ: " << std::setw(3) << static_cast<int>(thetaDeg) << "°"
                    << " │ Φ: " << std::setw(3) << static_cast<int>(phiDeg) << "°"
                    << " │ RT: " << std::fixed << std::setprecision(2) << rtKernelMs << "ms"
                    << " │ PO: " << poKernelMs << "ms"
                    << " │ Total: " << totalIterMs << "ms"
                    << "\n";
            }    

            totalIterations++;
        } // End of Phi loop
    } // End of Theta loop

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
    double totalTime = double(totalSweepEnd - totalSweepStart) / CLOCKS_PER_SEC;
    printEndSeparator();

    // ========== Print detailed performance summary ==========
    printSeparator("SWEEP PERFORMANCE BREAKDOWN");
    printKv("Ray Tracing Kernel (total)", totalRtKernelMs, "ms");
    printKv("PO Integration Kernel (total)", totalPoKernelMs, "ms");
    printKv("Memset (total)", totalMemsetMs, "ms");
    printKv("Memcpy D2H (total)", totalMemcpyMs, "ms");
    printKv("File Writes (total)", totalFileWriteMs, "ms");
    if (config.showHitStats) {
        printKv("Stats Computation (total)", totalStatsMs, "ms");
    }
    printKv("─────────────────", "");
    printKv("GPU Time (kernels only)", totalRtKernelMs + totalPoKernelMs, "ms");
    printKv("Total Sweep Wall Time", totalTime * 1000.0, "ms");
    printEndSeparator();

    printSeparator("PER-ITERATION AVERAGES");
    printKv("Ray Tracing Kernel", totalRtKernelMs / totalIterations, "ms");
    printKv("PO Integration Kernel", totalPoKernelMs / totalIterations, "ms");
    printKv("Memset", totalMemsetMs / totalIterations, "ms");
    printKv("Memcpy D2H", totalMemcpyMs / totalIterations, "ms");
    printKv("File Write", totalFileWriteMs / totalIterations, "ms");
    printKv("─────────────────", "");
    printKv("Total Points", totalIterations);
    printKv("Average Time/Point", totalTime * 1000.0 / totalIterations, "ms");
    printEndSeparator();

#ifndef USE_HIP
    nvtxRangePop();
#endif

    return sweepResults{totalIterations, totalTime};
}
