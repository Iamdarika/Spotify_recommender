#ifndef SONG_H
#define SONG_H

#include <string>

struct Song {
    std::string track_id;
    std::string track_name;
    std::string artists;
    int genre_id;

    Song() : genre_id(-1) {}
};

#endif // SONG_H

