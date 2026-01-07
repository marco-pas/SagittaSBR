#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <map>
#include <string>

std::map<std::string, float> loadConfig(const std::string& filename);

#endif
