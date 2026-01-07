/*
 * Implementation of formatted printing utilities.
 */

#include "printing_utils.h"
#include <iostream>
#include <iomanip>

void printSeparator(const std::string& title) {
    std::cerr << "\n┌── " << std::left << std::setw(46) << title << " ┐\n";
}

void printEndSeparator() {
    std::cerr << "└──────────────────────────────────────────────────┘\n";
}

template<typename T>
void printKV(const std::string& key, const T& value, const std::string& unit) {
    std::cerr << "│ " << std::left << std::setw(20) << key << ": " << value << " " << unit << "\n";
}

// Explicit template instantiations for common types
template void printKV<int>(const std::string&, const int&, const std::string&);
template void printKV<float>(const std::string&, const float&, const std::string&);
template void printKV<double>(const std::string&, const double&, const std::string&);
template void printKV<std::string>(const std::string&, const std::string&, const std::string&);

std::string formatTime(double seconds) {
    if (seconds < 0.001) return std::to_string(seconds * 1e6) + " µs";
    if (seconds < 1.0) return std::to_string(seconds * 1000) + " ms";
    return std::to_string(seconds) + " s";
}