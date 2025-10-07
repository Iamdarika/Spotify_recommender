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
};

#endif // DATAMANAGER_H
