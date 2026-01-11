#ifndef APP_PRINT_HPP
#define APP_PRINT_HPP

#include <iomanip>
#include <iostream>
#include <string>

void printSeparator(const std::string& title = "");
void printEndSeparator();
std::string formatTime(double seconds);

template<typename T>
void printKv(const std::string& key, const T& value, const std::string& unit = "") {
    std::cerr << "│ " << std::left << std::setw(20) << key << ": " << value << " " << unit << "\n";
}

#endif
