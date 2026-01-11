#include "app/config.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace {
std::map<std::string, float> readConfigFile(const std::string& filename) {
    std::map<std::string, float> config;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << ". Using defaults." << std::endl;
        return config;
    }

    while (std::getline(file, line)) {
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        std::stringstream ss(line);
        std::string key;
        float value;
        if (ss >> key >> value) {
            config[key] = value;
        }
    }

    return config;
}
}

simulationConfig loadConfig(const std::string& filename, int argc, char** argv) {
    simulationConfig config;
    auto raw = readConfigFile(filename);

    if (raw.count("phi_start")) {
        config.phiStart = raw["phi_start"];
    }
    if (raw.count("phi_end")) {
        config.phiEnd = raw["phi_end"];
    }
    if (raw.count("phi_samples")) {
        config.phiSamples = static_cast<int>(raw["phi_samples"]);
    }
    if (raw.count("theta_start")) {
        config.thetaStart = raw["theta_start"];
    }
    if (raw.count("theta_end")) {
        config.thetaEnd = raw["theta_end"];
    }
    if (raw.count("theta_samples")) {
        config.thetaSamples = static_cast<int>(raw["theta_samples"]);
    }
    if (raw.count("grid_size")) {
        config.gridSize = raw["grid_size"];
    }
    if (raw.count("nx")) {
        config.nx = static_cast<int>(raw["nx"]);
    }
    if (raw.count("ny")) {
        config.ny = static_cast<int>(raw["ny"]);
    }
    if (raw.count("max_bounces")) {
        config.maxBounces = static_cast<int>(raw["max_bounces"]);
    }
    if (raw.count("reflection_const")) {
        config.reflectionConst = raw["reflection_const"];
    }

    if (argc > 1) {
        config.freq = std::atof(argv[1]);
        std::cout << "Using frequency from command line: " << config.freq << " Hz\n";
    } else if (raw.count("freq")) {
        config.freq = raw["freq"];
        std::cout << "Using frequency from config file: " << config.freq << " Hz\n";
    } else {
        std::cout << "Using default frequency: " << config.freq << " Hz\n";
    }

    return config;
}
