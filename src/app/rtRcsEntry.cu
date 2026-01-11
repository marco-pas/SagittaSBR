#include "app/rtRcsEntry.hpp"

#include <fstream>
#include <iostream>

#include "app/config.hpp"
#include "app/deviceBuffers.hpp"
#include "app/print.hpp"
#include "cuda/cudaUtils.cuh"
#include "scene/world.cuh"
#include "sim/sweep.hpp"

int runRcsApp(int argc, char** argv) {
    std::cerr << "╔═══════════════════════════════════════════════════╗\n";
    std::cerr << "║  MONOSTATIC RCS SWEEP - SHOOTING & BOUNCING RAYS  ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════╝\n";

    printGpuInfo();

    simulationConfig config = loadConfig("config.txt", argc, argv);
    int rayCount = config.nx * config.ny;

    printSeparator("MEMORY ALLOCATION");
    deviceBuffers buffers;
    allocateDeviceBuffers(buffers, rayCount);

    hitable** deviceList = nullptr;
    hitable** deviceWorld = nullptr;
    checkCudaErrors(cudaMalloc(&deviceList, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&deviceWorld, sizeof(hitable*)));
    createWorld<<<1, 1>>>(deviceList, deviceWorld);
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "│  Managed memory and world setup complete.\n";
    printEndSeparator();

    std::ofstream outFile("rcs_results.csv");
    outFile << "# Frequency: " << config.freq << "\n# Grid: " << config.nx << "x" << config.ny << "\n";
    outFile << "# GridSize: " << config.gridSize << "\n# ThetaSamples: " << config.thetaSamples << "\n";
    outFile << "# PhiSamples: " << config.phiSamples << "\n" << "theta,phi,rcs_m2,rcs_dbsm\n";

    runSweep(config, buffers, deviceWorld, outFile);

    printSeparator("CLEANUP");
    freeWorld<<<1, 1>>>(deviceList, deviceWorld);
    checkCudaErrors(cudaDeviceSynchronize());
    freeDeviceBuffers(buffers);
    checkCudaErrors(cudaFree(deviceList));
    checkCudaErrors(cudaFree(deviceWorld));
    std::cerr << "│  Device memory released. Success.\n";
    printEndSeparator();

    return 0;
}
