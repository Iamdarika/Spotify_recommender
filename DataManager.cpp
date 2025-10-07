#include "DataManager.h"
#include <fstream>
#include <iostream>
#include <map>

bool DataManager::preprocessData(const std::string& csvPath, const std::string& outputPath) {
    std::ifstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csvPath << std::endl;
        return false;
    }

    std::string headerLine;
    if (!std::getline(csvFile, headerLine)) {
        std::cerr << "Error: Empty CSV file" << std::endl;
        return false;
    }

    std::vector<std::string> headers = parseCSVLine(headerLine);
    std::cout << "CSV Columns: " << headers.size() << std::endl;
    return true;
}
