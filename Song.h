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

    // Write Song data to a binary stream
    void serialize(std::ostream& out) const {
        size_t len;

        len = track_id.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(track_id.c_str(), len);

        len = track_name.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(track_name.c_str(), len);

        len = artists.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(artists.c_str(), len);

        out.write(reinterpret_cast<const char*>(&genre_id), sizeof(genre_id));
        out.write(reinterpret_cast<const char*>(features), FEATURE_COUNT * sizeof(float));
    }

    // Read Song data from a binary stream
    void deserialize(std::istream& in) {
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

        in.read(reinterpret_cast<char*>(&genre_id), sizeof(genre_id));
        in.read(reinterpret_cast<char*>(features), FEATURE_COUNT * sizeof(float));
    }
};

#endif // SONG_H
