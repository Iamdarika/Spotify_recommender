#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cstring>
#include <algorithm>
#include "Song.h"
#include "DataManager.h"
#include "Recommender.h"

const std::string BINARY_DATA_FILE = "songs_data.bin";

void printUsage(const char* programName) {
    std::cout << "Music Recommendation Engine - Usage:\n" << std::endl;
    std::cout << "1. Preprocessing Mode:" << std::endl;
    std::cout << "   " << programName << " --preprocess <path_to_csv>" << std::endl;
    std::cout << "   Processes CSV file and creates binary data file.\n" << std::endl;
    
    std::cout << "2. Recommendation Mode (by song name):" << std::endl;
    std::cout << "   " << programName << " --song \"Song Name\" [-n N]" << std::endl;
    std::cout << "   Returns top N similar songs (default N=10).\n" << std::endl;
    
    std::cout << "3. Recommendation Mode (by track ID):" << std::endl;
    std::cout << "   " << programName << " --id \"track_id\" [-n N]" << std::endl;
    std::cout << "   Returns top N similar songs (default N=10).\n" << std::endl;
    
    std::cout << "Examples:" << std::endl;
    std::cout << "   " << programName << " --preprocess dataset.csv" << std::endl;
    std::cout << "   " << programName << " --song \"Bohemian Rhapsody\" -n 5" << std::endl;
    std::cout << "   " << programName << " --id \"3ade68b8e\" -n 10" << std::endl;
}

bool preprocessMode(const std::string& csvPath) {
    std::cout << "=== PREPROCESSING MODE ===" << std::endl;
    
    if (!DataManager::preprocessData(csvPath, BINARY_DATA_FILE)) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return false;
    }
    
    std::cout << "\n✓ Preprocessing successful!" << std::endl;
    std::cout << "Binary data saved to: " << BINARY_DATA_FILE << std::endl;
    return true;
}

bool recommendationMode(const std::string& query, bool isTrackId, int topN) {
    std::cout << "=== RECOMMENDATION MODE ===" << std::endl;
    
    // Load preprocessed data
    std::vector<Song> songs;
    std::map<int, std::string> genreMap;
    
    if (!DataManager::loadData(BINARY_DATA_FILE, songs, genreMap)) {
        std::cerr << "Failed to load data. Have you run preprocessing?" << std::endl;
        return false;
    }
    
    // Initialize recommender
    Recommender recommender;
    if (!recommender.initialize(songs)) {
        std::cerr << "Failed to initialize recommender" << std::endl;
        return false;
    }
    
    // Get recommendations
    std::vector<int> recommendations;
    int queryIndex = -1;
    
    if (isTrackId) {
        std::cout << "\nSearching for track ID: " << query << std::endl;
        recommendations = recommender.recommend(query, topN);
        
        // Find query index for display
        for (size_t i = 0; i < songs.size(); ++i) {
            if (songs[i].track_id == query) {
                queryIndex = i;
                break;
            }
        }
    } else {
        std::cout << "\nSearching for song: " << query << std::endl;
        recommendations = recommender.recommendByName(query, topN);
        
        // Find query index for display
        for (size_t i = 0; i < songs.size(); ++i) {
            std::string lowerName = songs[i].track_name;
            std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
            std::string lowerQuery = query;
            std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);
            
            if (lowerName == lowerQuery || lowerName.find(lowerQuery) != std::string::npos) {
                queryIndex = i;
                break;
            }
        }
    }
    
    if (recommendations.empty()) {
        std::cerr << "No recommendations found. Please check the query." << std::endl;
        return false;
    }
    
    // Display query song info
    if (queryIndex >= 0) {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        std::cout << "Query Song:" << std::endl;
        std::cout << "  Title:   " << songs[queryIndex].track_name << std::endl;
        std::cout << "  Artist:  " << songs[queryIndex].artists << std::endl;
        std::cout << "  Genre:   " << genreMap[songs[queryIndex].genre_id] << std::endl;
        std::cout << "  ID:      " << songs[queryIndex].track_id << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    }
    
    // Display recommendations
    std::cout << "\nTop " << recommendations.size() << " Recommendations:\n" << std::endl;
    
    for (size_t i = 0; i < recommendations.size(); ++i) {
        int idx = recommendations[i];
        std::cout << (i + 1) << ". \"" << songs[idx].track_name << "\"" << std::endl;
        std::cout << "   Artist: " << songs[idx].artists << std::endl;
        std::cout << "   Genre:  " << genreMap[songs[idx].genre_id] << std::endl;
        std::cout << "   ID:     " << songs[idx].track_id << std::endl;
        
        if (i < recommendations.size() - 1) {
            std::cout << std::endl;
        }
    }
    
    std::cout << "\n✓ Recommendation complete!" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   High-Performance Music Recommendation Engine   ║" << std::endl;
    std::cout << "║          GPU-Accelerated with CUDA/cuBLAS         ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝\n" << std::endl;
    
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    std::string mode = argv[1];
    
    if (mode == "--preprocess") {
        if (argc < 3) {
            std::cerr << "Error: CSV path required for preprocessing mode" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        
        std::string csvPath = argv[2];
        return preprocessMode(csvPath) ? 0 : 1;
        
    } else if (mode == "--song" || mode == "--id") {
        if (argc < 3) {
            std::cerr << "Error: Song name or track ID required" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        
        std::string query = argv[2];
        bool isTrackId = (mode == "--id");
        int topN = 10; // Default
        
        // Check for -n option
        for (int i = 3; i < argc - 1; ++i) {
            if (strcmp(argv[i], "-n") == 0) {
                topN = std::atoi(argv[i + 1]);
                if (topN <= 0) {
                    std::cerr << "Error: Invalid value for -n (must be positive)" << std::endl;
                    return 1;
                }
                break;
            }
        }
        
        return recommendationMode(query, isTrackId, topN) ? 0 : 1;
        
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    return 0;
}

