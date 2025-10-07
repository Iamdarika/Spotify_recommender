#include "DataManager.h"
#include <sstream>
#include <algorithm>

std::string DataManager::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

bool DataManager::isValidNumber(const std::string& str) {
    if (str.empty()) return false;
    char* end;
    strtod(str.c_str(), &end);
    return end != str.c_str() && *end == '\0';
}

std::vector<std::string> DataManager::parseCSVLine(const std::string& line) {
    std::vector<std::string> result;
    std::string current;
    bool inQuotes = false;

    for (char c : line) {
        if (c == '"') inQuotes = !inQuotes;
        else if (c == ',' && !inQuotes) {
            result.push_back(trim(current));
            current.clear();
        } else current += c;
    }
    result.push_back(trim(current));
    return result;
}

