#ifndef SONG_H
#define SONG_H

#include <string>
#include <vector>
#include <iostream>

// Number of audio features used for similarity calculations
const int FEATURE_COUNT = 12;

/**
 * Song structure representing a track with its metadata and audio features.
 * 
 * Features used (in order):
 * danceability, energy, key, loudness, mode, speechiness, 
 * acousticness, instrumentalness, liveness, valence, tempo, genre_id
 */
struct Song {
    std::string track_id;
    std::string track_name;
    std::string artists;
    int genre_id;
    float features[FEATURE_COUNT];  // Normalized feature vector
    
    Song() : genre_id(-1) {
        for (int i = 0; i < FEATURE_COUNT; ++i) {
            features[i] = 0.0f;
        }
    }
    
    // Serialization: Write song to binary stream
    void serialize(std::ostream& out) const {
        // Write string lengths and data
        size_t len = track_id.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(track_id.c_str(), len);
        
        len = track_name.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(track_name.c_str(), len);
        
        len = artists.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(artists.c_str(), len);
        
        // Write genre_id
        out.write(reinterpret_cast<const char*>(&genre_id), sizeof(genre_id));
        
        // Write feature vector
        out.write(reinterpret_cast<const char*>(features), FEATURE_COUNT * sizeof(float));
    }
    
    // Deserialization: Read song from binary stream
    void deserialize(std::istream& in) {
        // Read strings
        size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        track_id.resize(len);
        in.read(&track_id[0], len);
        
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        track_name.resize(len);
        in.read(&track_name[0], len);
        
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        artists.resize(len);
        in.read(&artists[0], len);
        
        // Read genre_id
        in.read(reinterpret_cast<char*>(&genre_id), sizeof(genre_id));
        
        // Read feature vector
        in.read(reinterpret_cast<char*>(features), FEATURE_COUNT * sizeof(float));
    }
};

#endif // SONG_H
