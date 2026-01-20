#ifndef APP_CONFIG_HPP
#define APP_CONFIG_HPP

#include <string>

struct simulationConfig {
    float phiStart = 0.0f;
    float phiEnd = 180.0f;
    int phiSamples = 181;
    float thetaStart = 90.0f;
    float thetaEnd = 90.0f;
    int thetaSamples = 1;
    float gridSize = 3.0f;
    int nx = 400;
    int ny = 400;
    int tpbx = 16;
    int tpby = 16;
    int maxBounces = 20;
    float reflectionConst = 1.0f;
    float freq = 10.0e9f;
    bool showInfoGPU = true;
    bool showHitStats = true;
    std::string modelPath;
    float modelScale = 1.0f;

};

simulationConfig loadConfig(const std::string& filename, int argc, char** argv);

#endif
