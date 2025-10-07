#include "DataManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>

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
