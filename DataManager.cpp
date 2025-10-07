#include "DataManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <cmath>
#include <limits>

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
    std::cout << "Starting data preprocessing from: " << csvPath << std::endl;
    
    std::ifstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csvPath << std::endl;
        return false;
    }
    
    // Read header line
    std::string headerLine;
    if (!std::getline(csvFile, headerLine)) {
        std::cerr << "Error: Empty CSV file" << std::endl;
        return false;
    }
    
    // Remove BOM if present
    headerLine = removeBOM(headerLine);
    
    // Parse header to find column indices
    std::vector<std::string> headers = parseCSVLine(headerLine);
    std::map<std::string, int> columnMap;
    for (size_t i = 0; i < headers.size(); ++i) {
        columnMap[headers[i]] = i;
    }
    
    // Verify required columns exist
    std::vector<std::string> requiredCols = {
        "track_id", "track_name", "artists", "danceability", "energy", "key",
        "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "track_genre"
    };
    
    for (const auto& col : requiredCols) {
        if (columnMap.find(col) == columnMap.end()) {
            std::cerr << "Error: Required column '" << col << "' not found in CSV" << std::endl;
            return false;
        }
    }
    
    // Read all lines from CSV
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(csvFile, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    csvFile.close();
    
    std::cout << "Read " << lines.size() << " data rows from CSV" << std::endl;
    
    // Create temporary storage for parsed songs
    std::vector<Song> songs(lines.size());
    std::vector<bool> validSongs(lines.size(), false);
    std::map<std::string, int> genreToId;
    int nextGenreId = 0;
    
    // Store raw feature values for normalization
    std::vector<std::vector<float>> rawFeatures(lines.size(), std::vector<float>(FEATURE_COUNT - 1, 0.0f));
    
    // Feature column names (excluding genre which is categorical)
    std::vector<std::string> featureCols = {
        "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    };
    
    std::cout << "Parsing and validating songs..." << std::endl;
    
    
    // Count valid songs
    int validCount = 0;
    for (bool v : validSongs) {
        if (v) validCount++;
    }
    
    std::cout << "Valid songs: " << validCount << " out of " << lines.size() << std::endl;
    std::cout << "Unique genres: " << genreToId.size() << std::endl;
    
    if (validCount == 0) {
        std::cerr << "Error: No valid songs found in CSV" << std::endl;
        return false;
    }
    
    // Calculate min/max for each feature for normalization
    std::vector<float> minVals(FEATURE_COUNT - 1, std::numeric_limits<float>::max());
    std::vector<float> maxVals(FEATURE_COUNT - 1, std::numeric_limits<float>::lowest());
    
    for (size_t i = 0; i < lines.size(); ++i) {
        if (validSongs[i]) {
            for (int j = 0; j < FEATURE_COUNT - 1; ++j) {
                minVals[j] = std::min(minVals[j], rawFeatures[i][j]);
                maxVals[j] = std::max(maxVals[j], rawFeatures[i][j]);
            }
        }
    }
    
    std::cout << "Normalizing features..." << std::endl;

bool DataManager::loadData(const std::string& binaryPath, 
                          std::vector<Song>& songs,
                          std::map<int, std::string>& genreMap) {
    std::cout << "Loading preprocessed data from: " << binaryPath << std::endl;
    
    std::ifstream inFile(binaryPath, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open binary file: " << binaryPath << std::endl;
        return false;
    }
    
    // Read number of songs
    size_t numSongs;
    inFile.read(reinterpret_cast<char*>(&numSongs), sizeof(numSongs));
    
    // Read genre mapping
    size_t numGenres;
    inFile.read(reinterpret_cast<char*>(&numGenres), sizeof(numGenres));
    
    genreMap.clear();
    for (size_t i = 0; i < numGenres; ++i) {
        int genreId;
        size_t len;
        
        inFile.read(reinterpret_cast<char*>(&genreId), sizeof(genreId));
        inFile.read(reinterpret_cast<char*>(&len), sizeof(len));
        
        std::string genreName(len, '\0');
        inFile.read(&genreName[0], len);
        
        genreMap[genreId] = genreName;
    }
    
    // Read songs
    songs.clear();
    songs.resize(numSongs);
    
    for (size_t i = 0; i < numSongs; ++i) {
        songs[i].deserialize(inFile);
    }
    
    inFile.close();
    
    std::cout << "Loaded " << numSongs << " songs and " << numGenres << " genres." << std::endl;
    
    return true;
}
