// DataManager: Handles preprocessing of CSV song data into normalized binary format and loading it back into memory.
// Useful for efficiently storing, normalizing, and quickly loading large song datasets for analysis or ML tasks.

#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <string>
#include <vector>
#include <map>
#include "Song.h"

/**
 * DataManager handles preprocessing of raw CSV song data into 
 * normalized binary format using OpenMP parallelization.
 */
class DataManager {
public:
    /**
     * Process a CSV file containing song data and save to binary format.
     * 
     * @param csvPath Path to input CSV file
     * @param outputPath Path to output binary file
     * @return true if successful, false otherwise
     */
    static bool preprocessData(const std::string& csvPath, const std::string& outputPath);
    
    /**
     * Load preprocessed song data from binary file.
     * 
     * @param binaryPath Path to binary data file
     * @param songs Output vector to store loaded songs
     * @param genreMap Output map of genre_id to genre_name
     * @return true if successful, false otherwise
     */
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
