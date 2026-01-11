#ifndef SIM_SWEEP_HPP
#define SIM_SWEEP_HPP

#include <iosfwd>

#include "app/config.hpp"
#include "app/deviceBuffers.hpp"
#include "RT/hitable.h"

struct sweepResults {
    int totalIterations = 0;
    double totalTimeSeconds = 0.0;
};

sweepResults runSweep(const simulationConfig& config, deviceBuffers& buffers,
                      hitable** deviceWorld, std::ostream& outFile);

#endif
