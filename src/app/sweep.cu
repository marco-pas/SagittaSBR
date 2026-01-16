// actual simulation sweep

#include "sim/sweep.hpp"

#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "app/print.hpp"
#include "cuda/cudaUtils.cuh"
#include "cuda/poKernels.cuh"
#include "cuda/rtKernels.cuh"


sweepResults runSweep(const simulationConfig& config, deviceBuffers& buffers,
                      const bvhGpuData& bvhData, std::ostream& outFile) {

    // physical params
    const float c0 = 299792458.0f;
    const float lambda = c0 / config.freq;
    const float k = 2.0f * M_PI / lambda;
    const float rayArea = (config.gridSize * config.gridSize) / (config.nx * config.ny);
    const int nRays = config.nx * config.ny;
    const float distanceOffset = 10.0f;

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
    printEndSeparator();

    dim3 threads(config.tpbx, config.tpby);
    dim3 blocks(
        (config.nx + config.tpbx - 1) / config.tpbx, 
        (config.ny + config.tpby) / config.tpby
    );

    printSeparator("EXECUTING SWEEP");
    clock_t totalSweepStart = clock();
    int totalIterations = 0;

    for (int t = 0; t < config.thetaSamples; ++t) {
        float thetaDeg = config.thetaStart + (config.thetaSamples > 1
            ? t * (config.thetaEnd - config.thetaStart) / (config.thetaSamples - 1)
            : 0);
        float thetaRad = thetaDeg * M_PI / 180.0f;

        for (int p = 0; p < config.phiSamples; ++p) {
            clock_t iterStart = clock();

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

            buffers.accum->x = 0.0f;
            buffers.accum->y = 0.0f;

            launchRaysMultiBounce<<<blocks, threads>>>(
                buffers.hitPos, buffers.hitNormal, buffers.hitDist, buffers.hitCount,
                config.nx, config.ny, llc, uVec, vVec, rayDir,
                bvhData.triangles, bvhData.nodes, bvhData.rootIndex,
                config.maxBounces);
            checkCudaErrors(cudaDeviceSynchronize());

            integratePoMultiBounce<<<(nRays + 255) / 256, 256>>>(
                buffers.hitPos, buffers.hitNormal, buffers.hitDist, buffers.hitCount,
                nRays, k, rayDir, rayArea, buffers.accum, config.reflectionConst);
            checkCudaErrors(cudaDeviceSynchronize());

            float sigma = 4.0f * M_PI * (buffers.accum->x * buffers.accum->x
                         + buffers.accum->y * buffers.accum->y);
            float rcsDbsm = 10.0f * log10f(fmaxf(sigma, 1e-10f));

            outFile << thetaDeg << "," << phiDeg << "," << sigma << "," << rcsDbsm << "\n";

            clock_t iterEnd = clock();
            double iterTime = double(iterEnd - iterStart) / CLOCKS_PER_SEC;

            if (config.showHitStats) {
                // can be done with a kernel, but minor
                int maxHits = 0;
                int zeroCount = 0;
                long long sumHits = 0;
                int nonZeroCount = 0;

                for (int n = 0; n < nRays; n++) {
                    int val = buffers.hitCount[n];
                    if (val == 0) {
                        zeroCount++;
                    } else {
                        if (val > maxHits) maxHits = val;
                        sumHits += val;
                        nonZeroCount++;
                    }
                }

                float avgHits = (nonZeroCount > 0) ? static_cast<float>(sumHits) / nonZeroCount : 0.0f;

                std::cerr << "[" << std::setw(3) << totalIterations + 1 << "/"
                    << config.phiSamples * config.thetaSamples << "]"
                    << " | Θ: " << std::setw(3) << static_cast<int>(thetaDeg) << "°"
                    << " | Φ: " << std::setw(3) << static_cast<int>(phiDeg) << "°"
                    << " | " << formatTime(iterTime)
                    << " | Hit %: " << std::setprecision(1)
                    << 100.0f - (100.0f * zeroCount / nRays) << "%"
                    << " | Avg of bounces: " << std::fixed << std::setprecision(2) << avgHits
                    << " | Max bounces: " << maxHits << " (" << config.maxBounces << ")"
                    << "\n";
            } else {
                std::cerr << "[" << std::setw(3) << totalIterations + 1 << "/"
                    << config.phiSamples * config.thetaSamples << "]"
                    << " | Θ: " << std::setw(3) << static_cast<int>(thetaDeg) << "°"
                    << " | Φ: " << std::setw(3) << static_cast<int>(phiDeg) << "°"
                    << " | " << formatTime(iterTime) << "\n";
            }    

            totalIterations++;
        } // End of Phi loop
    } // End of Theta loop

    clock_t totalSweepEnd = clock();
    double totalTime = double(totalSweepEnd - totalSweepStart) / CLOCKS_PER_SEC;
    printEndSeparator();

    printSeparator("PERFORMANCE SUMMARY");
    printKv(" Total Sweep Time", formatTime(totalTime));
    printKv(" Total Points", totalIterations);
    printKv(" Average Time/Point", formatTime(totalTime / totalIterations));
    printEndSeparator();

    return sweepResults{totalIterations, totalTime};
}
