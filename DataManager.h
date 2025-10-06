#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <string>

class DataManager {
public:
    static bool preprocessData(const std::string& csvPath, const std::string& outputPath);
    static bool loadData(const std::string& binaryPath);
};

#endif // DATAMANAGER_H

