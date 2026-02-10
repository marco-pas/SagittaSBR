#ifndef APP_CONFIG_HPP
#define APP_CONFIG_HPP

#include <string>
#include "RT/precision.h"

struct simulationConfig {
    Real phiStart = REAL_CONST(0.0);
    Real phiEnd = REAL_CONST(180.0);
    int phiSamples = 181;
    Real thetaStart = REAL_CONST(90.0);
    Real thetaEnd = REAL_CONST(90.0);
    int thetaSamples = 1;
    Real gridSize = REAL_CONST(10.0);
    int nx = 5000;
    int ny = 5000;
    int tpbx = 16;
    int tpby = 16;
    int maxBounces = 20;
    Real reflectionConst = REAL_CONST(1.0);
    Real freq = REAL_CONST(10.0e9);
    bool showInfoGPU = true;
    bool showHitStats = true;
    std::string modelPath;
    Real modelScale = REAL_CONST(1.0);

};

simulationConfig loadConfig(const std::string& filename, int argc, char** argv);

#endif
