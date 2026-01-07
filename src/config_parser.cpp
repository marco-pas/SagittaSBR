/*
 * Implementation of configuration file parser.
 */

#include "config_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::map<std::string, float> loadConfig(const std::string& filename) {
    std::map<std::string, float> config;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << ". Using defaults." << std::endl;
        return config;
    }

    while (std::getline(file, line)) {
        // Strip out comments (anything after #)
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        // Extract key and value from the remaining string
        std::stringstream ss(line);
        std::string key;
        float value;
        if (ss >> key >> value) {
            config[key] = value;
        }
    }
    
    file.close();
    return config;
}
