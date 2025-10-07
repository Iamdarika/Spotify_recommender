#include "DataManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>

// Remove UTF-8 BOM if present
std::string removeBOM(const std::string& str) {
    if (str.size() >= 3 && 
        static_cast<unsigned char>(str[0]) == 0xEF && 
        static_cast<unsigned char>(str[1]) == 0xBB && 
        static_cast<unsigned char>(str[2]) == 0xBF) {
        return str.substr(3);
    }
    return str;
}

// Convert musical key notation to number (0-11)
int keyToNumber(const std::string& key) {
    std::string upperKey = key; // make a copy of the input string.
    std::transform(upperKey.begin(), upperKey.end(), upperKey.begin(), ::toupper);
    
    if (upperKey == "C") return 0;
    if (upperKey == "C#" || upperKey == "DB") return 1;
    if (upperKey == "D") return 2;
    if (upperKey == "D#" || upperKey == "EB") return 3;
    if (upperKey == "E") return 4;
    if (upperKey == "F") return 5;
    if (upperKey == "F#" || upperKey == "GB") return 6;
    if (upperKey == "G") return 7;
    if (upperKey == "G#" || upperKey == "AB") return 8;
    if (upperKey == "A") return 9;
    if (upperKey == "A#" || upperKey == "BB") return 10;
    if (upperKey == "B") return 11;
    
    return -1; // Invalid key
}

// Convert mode string to number (Major=1, Minor=0)
int modeToNumber(const std::string& mode) {
    std::string lowerMode = mode;
    std::transform(lowerMode.begin(), lowerMode.end(), lowerMode.begin(), ::tolower);
    
    if (lowerMode == "major" || lowerMode == "1") return 1;
    if (lowerMode == "minor" || lowerMode == "0") return 0;
    
    return -1; // Invalid mode
}

// Trim leading and trailing whitespace
std::string DataManager::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}
// Check if a string is a valid number
bool DataManager::isValidNumber(const std::string& str) {
    if (str.empty()) return false;
    char* end;
    strtod(str.c_str(), &end);
    return end != str.c_str() && *end == '\0'; //at least some characters were converted and entire string was processed
}

std::vector<std::string> DataManager::parseCSVLine(const std::string& line) {
    std::vector<std::string> result;
    std::string current;
    bool inQuotes = false;
    
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            result.push_back(trim(current));
            current.clear();
        } else {
            current += c;
        }
    }
    result.push_back(trim(current));
    
    return result;
}

bool DataManager::preprocessData(const std::string& csvPath, const std::string& outputPath) {
    std::ifstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file." << std::endl;
        return false;
    }

    std::string headerLine;
    std::getline(csvFile, headerLine);
    auto headers = parseCSVLine(headerLine);

    std::vector<std::string> required = {
        "track_id", "track_name", "artists", "danceability", "energy",
        "key", "loudness", "mode", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "track_genre"
    };

    std::map<std::string, int> columnMap;
    for (size_t i = 0; i < headers.size(); ++i) columnMap[headers[i]] = i;

    for (auto& c : required)
        if (columnMap.find(c) == columnMap.end()) {
            std::cerr << "Missing column: " << c << std::endl;
            return false;
        }

    std::vector<Song> songs;
    std::string line;
    while (std::getline(csvFile, line)) {
        auto fields = parseCSVLine(line);
        if (fields.size() < headers.size()) continue;

        Song song;
        song.track_id = fields[columnMap["track_id"]];
        song.track_name = fields[columnMap["track_name"]];
        song.artists = fields[columnMap["artists"]];
        songs.push_back(song);
    }

    std::cout << "Parsed " << songs.size() << " songs." << std::endl;
    return true;
}
