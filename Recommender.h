#ifndef RECOMMENDER_H
#define RECOMMENDER_H

#include <string>
#include <vector>
#include <map>
#include "Song.h"

/**
 * Structure to hold a recommendation result
 */
struct Recommendation {
    int songIndex;
    float similarity;
    
    Recommendation() : songIndex(-1), similarity(0.0f) {}
    Recommendation(int idx, float sim) : songIndex(idx), similarity(sim) {}
    
    bool operator<(const Recommendation& other) const {
        return similarity > other.similarity; // For max-heap
    }
};

/**
 * GPU-accelerated recommendation engine using CUDA and cuBLAS.
 * Calculates cosine similarity between songs on the GPU.
 */
class Recommender {
public:
    Recommender();
    ~Recommender();
    
    /**
     * Initialize the recommender with song data.
     * Transfers feature vectors to GPU memory.
     * 
     * @param songs Vector of all songs
     * @return true if successful, false otherwise
     */
    bool initialize(const std::vector<Song>& songs);
    
    /**
     * Get recommendations for a song by track_id.
     * 
     * @param trackId The track ID to find recommendations for
     * @param topN Number of recommendations to return
     * @return Vector of recommended song indices (sorted by similarity)
     */
    std::vector<int> recommend(const std::string& trackId, int topN);
    
    /**
     * Get recommendations for a song by track name.
     * 
     * @param trackName The track name to find recommendations for
     * @param topN Number of recommendations to return
     * @return Vector of recommended song indices (sorted by similarity)
     */
    std::vector<int> recommendByName(const std::string& trackName, int topN);
    
    /**
     * Get recommendations for a song by index.
     * 
     * @param songIndex Index of the song in the database
     * @param topN Number of recommendations to return
     * @return Vector of recommended song indices (sorted by similarity)
     */
    std::vector<int> recommendByIndex(int songIndex, int topN);
    
    /**
     * Check if the recommender is initialized.
     */
    bool isInitialized() const { return initialized; }
    
    /**
     * Get the number of songs in the database.
     */
    int getSongCount() const { return numSongs; }
    
private:
    bool initialized;
    int numSongs;
    
    // CPU data
    std::vector<Song> songDatabase;
    
    // GPU data
    float* d_features;          // Device memory for all feature vectors (numSongs x FEATURE_COUNT)
    float* d_queryFeature;      // Device memory for query feature vector
    float* d_similarities;      // Device memory for similarity scores
    
    // cuBLAS handle
    void* cublasHandle;  // Using void* to avoid including cublas in header
    
    /**
     * Calculate cosine similarities on GPU using cuBLAS.
     * 
     * @param queryIndex Index of the query song
     * @param similarities Output array for similarity scores (host memory)
     */
    void calculateSimilarities(int queryIndex, float* similarities);
    
    /**
     * Find song index by track ID.
     */
    int findSongByTrackId(const std::string& trackId) const;
    
    /**
     * Find song index by track name (case-insensitive).
     */
    int findSongByName(const std::string& trackName) const;
    
    /**
     * Convert string to lowercase for case-insensitive comparison.
     */
    static std::string toLower(const std::string& str);
};

#endif // RECOMMENDER_H

