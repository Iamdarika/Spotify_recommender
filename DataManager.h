#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <string>
#include <vector>
#include <map>
#include "Song.h"

class DataManager {
public:
    // Process a CSV file and save the data in binary format
    static bool preprocessData(const std::string& csvPath, const std::string& outputPath);
    
    // Load preprocessed data from binary file
    static bool loadData(const std::string& binaryPath,
                        std::vector<Song>& songs,
                        std::map<int, std::string>& genreMap);

private:
    // Helper to parse a CSV line considering quoted fields
    static std::vector<std::string> parseCSVLine(const std::string& line);

    // Helper to trim whitespace from strings
    static std::string trim(const std::string& str);

    // Helper to check if string is a valid number
    static bool isValidNumber(const std::string& str);
};

#endif // DATAMANAGER_H

