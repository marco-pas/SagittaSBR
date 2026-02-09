#include "app/config.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace {
std::map<std::string, double> readConfigFile(const std::string& filename) {
    std::map<std::string, double> config;
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
        double value;
        if (ss >> key >> value) {
            config[key] = value;
        }
    }

    return config;
}
}

simulationConfig loadConfig(const std::string& filename, int argc, char** argv) {
    // Allow --config <path> to override the default config file path
    std::string configPath = filename;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
            break;
        }
    }

    simulationConfig config;
    auto raw = readConfigFile(configPath);

    if (raw.count("phi_start")) {
        config.phiStart = static_cast<Real>(raw["phi_start"]);
    }
    if (raw.count("phi_end")) {
        config.phiEnd = static_cast<Real>(raw["phi_end"]);
    }
    if (raw.count("phi_samples")) {
        config.phiSamples = static_cast<int>(raw["phi_samples"]);
    }
    if (raw.count("theta_start")) {
        config.thetaStart = static_cast<Real>(raw["theta_start"]);
    }
    if (raw.count("theta_end")) {
        config.thetaEnd = static_cast<Real>(raw["theta_end"]);
    }
    if (raw.count("theta_samples")) {
        config.thetaSamples = static_cast<int>(raw["theta_samples"]);
    }
    if (raw.count("grid_size")) {
        config.gridSize = static_cast<Real>(raw["grid_size"]);
    }
    if (raw.count("n_x")) {
        config.nx = static_cast<int>(raw["n_x"]);
    }
    if (raw.count("n_y")) {
        config.ny = static_cast<int>(raw["n_y"]);
    }
    if (raw.count("tpb_x")) {
        config.tpbx = static_cast<int>(raw["tpb_x"]);
    }
    if (raw.count("tpb_y")) {
        config.tpby = static_cast<int>(raw["tpb_y"]);
    }
    if (raw.count("max_bounces")) {
        config.maxBounces = static_cast<int>(raw["max_bounces"]);
    }
    if (raw.count("reflection_const")) {
        config.reflectionConst = static_cast<Real>(raw["reflection_const"]);
    }

    if (raw.count("show_info_GPU")) {
        config.showInfoGPU = static_cast<bool>(raw["show_info_GPU"]);

    }
    if (raw.count("show_hit_stats")) {
        config.showHitStats = static_cast<bool>(raw["show_hit_stats"]);
    }

    bool freqFromConfig = false;
    if (raw.count("freq")) {
        config.freq = static_cast<Real>(raw["freq"]);
        freqFromConfig = true;
    }

    bool freqFromCli = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --model requires a file path.\n";
                std::exit(1);
            }
            config.modelPath = argv[++i];
            continue;
        }

        if (arg == "--model-scale") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --model-scale requires a numeric value.\n";
                std::exit(1);
            }
            config.modelScale = static_cast<Real>(std::atof(argv[++i]));
            continue;
        }

        if (arg == "--config") {
            // Already handled above; skip the value
            ++i;
            continue;
        }

        if (!arg.empty() && arg[0] != '-') {
            if (!freqFromCli) {
                config.freq = static_cast<Real>(std::atof(arg.c_str()));
                freqFromCli = true;
            }
            continue;
        }

        std::cerr << "Warning: Unrecognized argument '" << arg << "'.\n";
    }

    if (freqFromCli) {
        std::cout << "Using frequency from command line: " << config.freq << " Hz\n";
    } else if (freqFromConfig) {
        std::cout << "Using frequency from config file: " << config.freq << " Hz\n";
    } else {
        std::cout << "Using default frequency: " << config.freq << " Hz\n";
    }

    return config;
}
