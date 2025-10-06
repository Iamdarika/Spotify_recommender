#ifndef SONG_H
#define SONG_H

#include <string>
#include <iostream>

// Number of audio features used for similarity calculations
const int FEATURE_COUNT = 12;

struct Song {
    std::string track_id;
    std::string track_name;
    std::string artists;
    int genre_id;
    float features[FEATURE_COUNT];  // Feature vector for audio attributes

    Song() : genre_id(-1) {
        for (int i = 0; i < FEATURE_COUNT; ++i) {
            features[i] = 0.0f;
        }
    }

    void printInfo() const {
        std::cout << "Track: " << track_name 
                  << " | Artist: " << artists 
                  << " | Genre ID: " << genre_id << std::endl;
    }
};

#endif // SONG_H
