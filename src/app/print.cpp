#include "app/print.hpp"

#include <sstream>

// Print formatting for nice output

void printSeparator(const std::string& title) {
    std::cerr << "\n┌── " << std::left << std::setw(46) << title << " ┐\n";
}

void printEndSeparator() {
    std::cerr << "└──────────────────────────────────────────────────┘\n";
}

std::string formatTime(double seconds) {
    std::ostringstream out;
    if (seconds < 0.001) {
        out << seconds * 1e6 << " µs";
    } else if (seconds < 1.0) {
        out << seconds * 1000 << " ms";
    } else {
        out << seconds << " s";
    }
    return out.str();
}
