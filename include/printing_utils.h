/*
 * Formatted console output utilities for simulation progress and statistics.
 * Provides functions for printing separators, key-value pairs, and time formatting.
 */

#ifndef PRINTING_UTILS_H
#define PRINTING_UTILS_H

#include <string>

// Print a separator with optional title
void printSeparator(const std::string& title = "");

// Print end separator
void printEndSeparator();

// Print key-value pair with optional unit
template<typename T>
void printKV(const std::string& key, const T& value, const std::string& unit = "");

// Format time in appropriate units (µs, ms, or s)
std::string formatTime(double seconds);

#endif // PRINTING_UTILS_H
